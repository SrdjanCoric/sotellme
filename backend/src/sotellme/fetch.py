import html
import ipaddress
import json
import re
from collections.abc import Callable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

import httpx

from sotellme.extraction import MAX_DOCUMENT_BYTES

FETCH_TIMEOUT_SECONDS = 15.0

MAX_PAGE_CHARS = 10_000

PASTE_FALLBACK = "Copy the page and paste the posting text instead."

_INVISIBLE_ELEMENTS = frozenset({"script", "style", "noscript", "head", "template"})

_LD_JSON_TYPE = "application/ld+json"


class PostingFetchError(Exception):
    pass


class _VisibleTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []
        self.ld_json_blocks: list[str] = []
        self._skip_depth = 0
        self._in_ld_json = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _INVISIBLE_ELEMENTS:
            self._skip_depth += 1
        if tag == "script" and dict(attrs).get("type") == _LD_JSON_TYPE:
            self._in_ld_json = True

    def handle_endtag(self, tag: str) -> None:
        if tag in _INVISIBLE_ELEMENTS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "script":
            self._in_ld_json = False

    def handle_data(self, data: str) -> None:
        if self._in_ld_json and data.strip():
            self.ld_json_blocks.append(data)
        elif not self._skip_depth and data.strip():
            self.chunks.append(" ".join(data.split()))


def _parse_page(markup: str) -> _VisibleTextExtractor:
    extractor = _VisibleTextExtractor()
    extractor.feed(markup)
    extractor.close()
    return extractor


def html_to_text(markup: str) -> str:
    return "\n".join(_parse_page(markup).chunks)


def _ld_json_candidates(block: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return []
    items = data if isinstance(data, list) else [data]
    candidates: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            candidates.append(item)
            graph = item.get("@graph")
            if isinstance(graph, list):
                candidates.extend(entry for entry in graph if isinstance(entry, dict))
    return candidates


def _job_posting_text(page: _VisibleTextExtractor) -> str:
    for block in page.ld_json_blocks:
        for candidate in _ld_json_candidates(block):
            if candidate.get("@type") != "JobPosting":
                continue
            organization = candidate.get("hiringOrganization")
            company = organization.get("name") if isinstance(organization, dict) else None
            sections = [
                str(candidate.get("title") or ""),
                str(company or ""),
                html_to_text(html.unescape(str(candidate.get("description") or ""))),
            ]
            text = "\n\n".join(section for section in sections if section.strip())
            if text.strip():
                return text
    return ""


def _normalize_url(value: str) -> str:
    url = value.strip()
    if url.lower().startswith("www."):
        url = f"https://{url}"
    return url


_WORKABLE_JOB_PATH = re.compile(r"^/([^/]+)/j/([A-Za-z0-9]+)/?$")


def _workable_api_url(url: str) -> str | None:
    parts = urlsplit(url)
    if (parts.hostname or "").lower() != "apply.workable.com":
        return None
    match = _WORKABLE_JOB_PATH.match(parts.path)
    if match is None:
        return None
    account, shortcode = match.groups()
    return f"https://apply.workable.com/api/v2/accounts/{account}/jobs/{shortcode}"


def _workable_posting_text(body: str) -> str:
    try:
        job = json.loads(body)
    except json.JSONDecodeError:
        return ""
    if not isinstance(job, dict):
        return ""
    sections = [str(job.get("title") or "")]
    sections.extend(
        html_to_text(str(job.get(field) or ""))
        for field in ("description", "requirements", "benefits")
    )
    return "\n\n".join(section for section in sections if section.strip())


def _download(
    url: str,
    transport: httpx.BaseTransport | None,
    request_hook: Callable[[httpx.Request], None] | None = None,
) -> tuple[str, str]:
    event_hooks = {"request": [request_hook]} if request_hook else {}
    with (
        httpx.Client(
            follow_redirects=True,
            timeout=FETCH_TIMEOUT_SECONDS,
            transport=transport,
            event_hooks=event_hooks,
        ) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        chunks: list[bytes] = []
        received = 0
        for chunk in response.iter_bytes():
            received += len(chunk)
            if received > MAX_DOCUMENT_BYTES:
                raise PostingFetchError(f"The page is larger than 5 MB. {PASTE_FALLBACK}")
            chunks.append(chunk)
        encoding = response.encoding or "utf-8"
        content_type = response.headers.get("content-type", "")
    return b"".join(chunks).decode(encoding, errors="replace"), content_type


def fetch_posting_text(value: str, transport: httpx.BaseTransport | None = None) -> str:
    url = _normalize_url(value)
    api_url = _workable_api_url(url)
    try:
        body, content_type = _download(api_url or url, transport)
    except httpx.HTTPStatusError as exc:
        raise PostingFetchError(
            f"The link answered with HTTP {exc.response.status_code}. {PASTE_FALLBACK}"
        ) from exc
    except httpx.HTTPError as exc:
        raise PostingFetchError(f"Could not fetch the link: {exc}. {PASTE_FALLBACK}") from exc
    if api_url is not None:
        text = _workable_posting_text(body)
    elif "html" in content_type:
        page = _parse_page(body)
        text = _job_posting_text(page) or "\n".join(page.chunks)
    else:
        text = body
    if not text.strip():
        raise PostingFetchError(
            f"No readable text found at that link; the page probably builds its content "
            f"with JavaScript in the browser. {PASTE_FALLBACK}"
        )
    return text


class ResearchFetchError(Exception):
    pass


def _refuse_unsafe_host(url: httpx.URL) -> None:
    if url.scheme not in ("http", "https"):
        raise ResearchFetchError(f"Fetching {url.scheme!r} links is refused.")
    host = url.host
    if not host:
        raise ResearchFetchError("A link without a host is refused.")
    if host == "localhost" or host.endswith(".localhost"):
        raise ResearchFetchError("Fetching localhost is refused.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if not address.is_global:
        raise ResearchFetchError(f"Fetching the private or local address {host} is refused.")


def fetch_research_page(url: str, transport: httpx.BaseTransport | None = None) -> str:
    def check_each_hop(request: httpx.Request) -> None:
        _refuse_unsafe_host(request.url)

    _refuse_unsafe_host(httpx.URL(url))
    try:
        body, content_type = _download(url, transport, request_hook=check_each_hop)
    except ResearchFetchError:
        raise
    except httpx.HTTPStatusError as exc:
        raise ResearchFetchError(
            f"The page answered with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        raise ResearchFetchError(f"Could not fetch the page: {exc}.") from exc
    text = html_to_text(body) if "html" in content_type else body
    if not text.strip():
        raise ResearchFetchError("No readable text found on the page.")
    return text[:MAX_PAGE_CHARS]

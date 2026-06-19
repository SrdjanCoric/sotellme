"""Fetch and extract readable text from job postings and research web pages."""

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
    """Raised when a job posting cannot be fetched or yields no readable text."""

    pass


class _VisibleTextExtractor(HTMLParser):
    """Collect visible text chunks and ld+json blocks while parsing HTML."""

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
    """Parse markup and return the extractor holding its visible text and ld+json blocks."""
    extractor = _VisibleTextExtractor()
    extractor.feed(markup)
    extractor.close()
    return extractor


def html_to_text(markup: str) -> str:
    """Extract visible text from HTML markup.

    Text inside script, style, noscript, head, and template elements is dropped.

    Args:
        markup: The HTML source to extract from.

    Returns:
        The visible text with one chunk per line.
    """
    return "\n".join(_parse_page(markup).chunks)


def _ld_json_candidates(block: str) -> list[dict[str, Any]]:
    """Parse an ld+json block into candidate dicts, including any @graph entries."""
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
    """Build posting text from the first JobPosting found in the page's ld+json blocks."""
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
    """Trim a URL and prefix ``https://`` when it starts with ``www.``."""
    url = value.strip()
    if url.lower().startswith("www."):
        url = f"https://{url}"
    return url


_WORKABLE_JOB_PATH = re.compile(r"^/([^/]+)/j/([A-Za-z0-9]+)/?$")


def _workable_api_url(url: str) -> str | None:
    """Return the Workable JSON API URL for a job page, or None if it is not one."""
    parts = urlsplit(url)
    if (parts.hostname or "").lower() != "apply.workable.com":
        return None
    match = _WORKABLE_JOB_PATH.match(parts.path)
    if match is None:
        return None
    account, shortcode = match.groups()
    return f"https://apply.workable.com/api/v2/accounts/{account}/jobs/{shortcode}"


def _workable_posting_text(body: str) -> str:
    """Build posting text from a Workable API JSON body's title and HTML fields."""
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
    """Stream a URL and return its decoded body and content type."""
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
    """Fetch a job posting link and return its readable text.

    Workable job links are read through the Workable JSON API; HTML pages prefer the
    JobPosting ld+json data and fall back to visible text; other content types are returned
    as-is.

    Args:
        value: The posting URL, possibly missing a scheme.
        transport: Optional httpx transport, mainly for testing.

    Returns:
        The posting text.

    Raises:
        PostingFetchError: If the link errors, is invalid, exceeds the size limit, or yields
            no readable text.
    """
    url = _normalize_url(value)
    try:
        api_url = _workable_api_url(url)
        body, content_type = _download(api_url or url, transport)
    except httpx.HTTPStatusError as exc:
        raise PostingFetchError(
            f"The link answered with HTTP {exc.response.status_code}. {PASTE_FALLBACK}"
        ) from exc
    except httpx.HTTPError as exc:
        raise PostingFetchError(f"Could not fetch the link: {exc}. {PASTE_FALLBACK}") from exc
    except (ValueError, httpx.InvalidURL) as exc:
        raise PostingFetchError(
            f"That doesn't look like a valid link: {exc}. {PASTE_FALLBACK}"
        ) from exc
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
    """Raised when a research page cannot be fetched safely or yields no readable text."""

    pass


def _refuse_unsafe_host(url: httpx.URL) -> None:
    """Raise ResearchFetchError for non-http(s), hostless, localhost, or private-IP URLs."""
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
    """Fetch a research page safely and return its readable text, length-capped.

    The host is checked before the request and on every redirect hop to refuse unsafe targets.
    HTML bodies are reduced to visible text; other content types are returned as-is. The
    result is truncated to the maximum page length.

    Args:
        url: The page URL to fetch.
        transport: Optional httpx transport, mainly for testing.

    Returns:
        The page text, truncated to the maximum page length.

    Raises:
        ResearchFetchError: If the host is unsafe, the request errors, the link is invalid, or
            the page yields no readable text.
    """

    def check_each_hop(request: httpx.Request) -> None:
        _refuse_unsafe_host(request.url)

    try:
        _refuse_unsafe_host(httpx.URL(url))
        body, content_type = _download(url, transport, request_hook=check_each_hop)
    except ResearchFetchError:
        raise
    except httpx.HTTPStatusError as exc:
        raise ResearchFetchError(
            f"The page answered with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        raise ResearchFetchError(f"Could not fetch the page: {exc}.") from exc
    except (ValueError, httpx.InvalidURL) as exc:
        raise ResearchFetchError(f"That doesn't look like a valid link: {exc}.") from exc
    text = html_to_text(body) if "html" in content_type else body
    if not text.strip():
        raise ResearchFetchError("No readable text found on the page.")
    return text[:MAX_PAGE_CHARS]

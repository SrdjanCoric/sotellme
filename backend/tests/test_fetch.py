import inspect

import httpx
import pytest

import sotellme.engine
import sotellme.fetch
import sotellme.flagger
import sotellme.interviewer
import sotellme.profile
import sotellme.role
from sotellme.extraction import MAX_DOCUMENT_BYTES
from sotellme.fetch import PostingFetchError, fetch_posting_text, html_to_text

PAGE = (
    "<html><head><title>Job</title><script>trackVisitors()</script>"
    "<style>.hidden { display: none }</style></head>"
    "<body><h1>Senior Backend Engineer</h1>"
    "<p>Acme builds billing software for veterinary clinics.</p></body></html>"
)


def serving(response: httpx.Response) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: response)


def test_html_to_text_keeps_visible_text_and_drops_scripts_and_styles() -> None:
    text = html_to_text(PAGE)

    assert "Senior Backend Engineer" in text
    assert "Acme builds billing software for veterinary clinics." in text
    assert "trackVisitors" not in text
    assert "display: none" not in text


def test_a_fetched_page_yields_its_posting_text() -> None:
    transport = serving(httpx.Response(200, html=PAGE))

    text = fetch_posting_text("https://jobs.acme.com/senior-backend", transport=transport)

    assert "Senior Backend Engineer" in text
    assert "trackVisitors" not in text


def test_a_plain_text_response_passes_through_unstripped() -> None:
    posting = "Senior Backend Engineer at Acme."
    transport = serving(httpx.Response(200, text=posting, headers={"content-type": "text/plain"}))

    assert fetch_posting_text("https://jobs.acme.com/raw", transport=transport) == posting


def test_a_bare_www_link_is_fetched_over_https() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.scheme == "https"
        return httpx.Response(200, html=PAGE)

    text = fetch_posting_text("www.acme.com/careers/123", transport=httpx.MockTransport(handler))

    assert "Senior Backend Engineer" in text


def test_an_http_error_says_paste_instead() -> None:
    transport = serving(httpx.Response(404))

    with pytest.raises(PostingFetchError, match=r"404.*paste the posting text"):
        fetch_posting_text("https://jobs.acme.com/gone", transport=transport)


def test_an_oversized_page_is_rejected() -> None:
    transport = serving(httpx.Response(200, content=b"x" * (MAX_DOCUMENT_BYTES + 1)))

    with pytest.raises(PostingFetchError, match="5 MB"):
        fetch_posting_text("https://jobs.acme.com/huge", transport=transport)


def test_an_empty_page_is_rejected() -> None:
    transport = serving(httpx.Response(200, html="<html><body></body></html>"))

    with pytest.raises(PostingFetchError, match="No readable text"):
        fetch_posting_text("https://jobs.acme.com/blank", transport=transport)


JOB_POSTING_PAGE = (
    "<html><head>"
    '<script type="application/ld+json">'
    '{"@context": "https://schema.org/", "@type": "JobPosting",'
    ' "title": "Staff Engineer",'
    ' "hiringOrganization": {"@type": "Organization", "name": "Acme"},'
    ' "description": "&lt;p&gt;Own the billing platform end to end.&lt;/p&gt;"}'
    "</script></head>"
    "<body><nav>Skip to main content</nav><p>People also viewed: Junior Cook</p></body></html>"
)


def test_a_job_posting_ld_json_block_is_preferred_over_page_chrome() -> None:
    transport = serving(httpx.Response(200, html=JOB_POSTING_PAGE))

    text = fetch_posting_text("https://www.linkedin.com/jobs/view/12345", transport=transport)

    assert "Staff Engineer" in text
    assert "Acme" in text
    assert "Own the billing platform end to end." in text
    assert "People also viewed" not in text


def test_a_page_without_a_job_posting_block_falls_back_to_visible_text() -> None:
    chrome = (
        '<html><head><script type="application/ld+json">'
        '{"@type": "Organization", "name": "Acme"}'
        "</script></head><body><p>Senior Backend Engineer at Acme.</p></body></html>"
    )
    transport = serving(httpx.Response(200, html=chrome))

    text = fetch_posting_text("https://jobs.acme.com/1", transport=transport)

    assert text == "Senior Backend Engineer at Acme."


def test_a_malformed_ld_json_block_falls_back_to_visible_text() -> None:
    broken = (
        '<html><head><script type="application/ld+json">{not json'
        "</script></head><body><p>Senior Backend Engineer at Acme.</p></body></html>"
    )
    transport = serving(httpx.Response(200, html=broken))

    text = fetch_posting_text("https://jobs.acme.com/2", transport=transport)

    assert text == "Senior Backend Engineer at Acme."


WORKABLE_JOB = {
    "title": "Product Engineer",
    "description": "<p>Acme is a remote-first company.</p>",
    "requirements": "<ul><li>Strong engineering fundamentals.</li></ul>",
    "benefits": "<p>Professional development.</p>",
}


def test_a_workable_job_link_is_read_through_the_public_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v2/accounts/acme-1/jobs/451EC658A2"
        return httpx.Response(200, json=WORKABLE_JOB)

    text = fetch_posting_text(
        "https://apply.workable.com/acme-1/j/451EC658A2/", transport=httpx.MockTransport(handler)
    )

    assert "Product Engineer" in text
    assert "Acme is a remote-first company." in text
    assert "Strong engineering fundamentals." in text
    assert "<p>" not in text


def test_a_non_job_workable_page_uses_the_plain_html_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/acme-1/"
        return httpx.Response(200, html=PAGE)

    text = fetch_posting_text(
        "https://apply.workable.com/acme-1/", transport=httpx.MockTransport(handler)
    )

    assert "Senior Backend Engineer" in text


def test_a_javascript_only_page_explains_itself() -> None:
    shell = "<html><head><script>render()</script></head><body></body></html>"
    transport = serving(httpx.Response(200, html=shell))

    with pytest.raises(PostingFetchError, match="JavaScript"):
        fetch_posting_text("https://jobs.acme.com/spa", transport=transport)


def test_the_fetcher_never_reads_the_environment() -> None:
    source = inspect.getsource(sotellme.fetch)

    assert "environ" not in source
    assert "getenv" not in source


def test_no_model_facing_module_can_reach_the_fetcher() -> None:
    for module in (
        sotellme.engine,
        sotellme.interviewer,
        sotellme.role,
        sotellme.profile,
        sotellme.flagger,
    ):
        source = inspect.getsource(module)
        assert "sotellme.fetch" not in source, f"{module.__name__} imports the fetcher"
        assert "fetch_posting_text" not in source, f"{module.__name__} can trigger a fetch"

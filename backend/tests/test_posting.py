from pathlib import Path

import pytest

from sotellme.fetch import PostingFetchError
from sotellme.posting import PostingInputError, resolve_posting_text


def test_pasted_text_passes_through() -> None:
    posting = "Senior Backend Engineer at Acme. You will own the billing platform."

    assert resolve_posting_text(posting) == posting


@pytest.mark.parametrize(
    "url",
    [
        "https://jobs.acme.com/senior-backend",
        "http://acme.com/careers/123",
        "www.acme.com/careers/123",
    ],
)
def test_a_url_is_dispatched_to_the_fetcher(url: str) -> None:
    fetched: list[str] = []

    def fake_fetch(value: str) -> str:
        fetched.append(value)
        return "Backend Engineer at Acme."

    assert resolve_posting_text(url, fetch=fake_fetch) == "Backend Engineer at Acme."
    assert fetched == [url]


def test_a_failed_fetch_surfaces_as_a_posting_input_error() -> None:
    def failing_fetch(value: str) -> str:
        raise PostingFetchError(
            "The link answered with HTTP 403. Copy the page and paste the posting text instead."
        )

    with pytest.raises(PostingInputError, match="paste the posting text"):
        resolve_posting_text("https://jobs.acme.com/blocked", fetch=failing_fetch)


def test_a_file_path_reads_the_file(tmp_path: Path) -> None:
    posting_file = tmp_path / "posting.txt"
    posting_file.write_text("Backend Engineer at Acme. Own the billing platform.")

    assert resolve_posting_text(str(posting_file)) == (
        "Backend Engineer at Acme. Own the billing platform."
    )


def test_an_empty_posting_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(PostingInputError, match="empty"):
        resolve_posting_text("   \n  ")

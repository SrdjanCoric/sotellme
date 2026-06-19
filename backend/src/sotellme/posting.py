"""Resolve job posting text from a URL, a file path, or inline text."""

from collections.abc import Callable
from pathlib import Path

from sotellme.extraction import CVInputError, extract_document_text
from sotellme.fetch import PostingFetchError, fetch_posting_text


class PostingInputError(Exception):
    """Raised when a job posting cannot be fetched, read, or is empty."""


def resolve_posting_text(value: str, fetch: Callable[[str], str] = fetch_posting_text) -> str:
    """Resolve job posting text from a URL, a file path, or inline text.

    If the value looks like a URL it is fetched; otherwise, if it names an existing
    file, the file's text is extracted; otherwise the value is treated as the posting
    text itself.

    Args:
        value: A URL, a file path, or the posting text.
        fetch: Callable used to fetch posting text from a URL.

    Returns:
        The resolved posting text.

    Raises:
        PostingInputError: If fetching or reading fails, or the resolved text is empty.
    """
    if value.strip().lower().startswith(("http://", "https://", "www.")):
        try:
            return fetch(value.strip())
        except PostingFetchError as exc:
            raise PostingInputError(str(exc)) from exc
    path = Path(value)
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if is_file:
        try:
            value = extract_document_text(path, label="posting")
        except CVInputError as exc:
            raise PostingInputError(str(exc)) from exc
    if not value.strip():
        raise PostingInputError("The job posting is empty. Paste the posting text or a file path.")
    return value

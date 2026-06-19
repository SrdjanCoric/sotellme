"""Extract plain text from CV and posting documents in PDF, markdown, or text form."""

import io
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

MAX_DOCUMENT_BYTES = 5 * 1024 * 1024


class CVInputError(Exception):
    """Raised when a document cannot be read or is in an unsupported format."""


def extract_cv_text(path: Path) -> str:
    """Extract text from a CV document.

    Args:
        path: Path to the CV file.

    Returns:
        The extracted document text.

    Raises:
        CVInputError: If the file is too large, unsupported, or unreadable.
    """
    return extract_document_text(path, label="CV")


def extract_document_text(path: Path, label: str) -> str:
    """Extract text from a document, detecting its format from the leading bytes.

    Reads PDF files via the PDF extractor and decodes plain text or markdown as UTF-8.
    Word documents (detected by the 'PK' zip signature) are rejected.

    Args:
        path: Path to the document file.
        label: Human-readable label for the document, used in error messages.

    Returns:
        The extracted document text.

    Raises:
        CVInputError: If the file exceeds the size limit, is a Word document, or
            cannot be decoded as UTF-8.
    """
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise CVInputError(f"The {label} file is larger than 5 MB. Trim it down and try again.")
    raw = path.read_bytes()
    if raw.startswith(b"%PDF"):
        return _extract_pdf_text(raw, label)
    if raw.startswith(b"PK"):
        raise CVInputError(
            "Word documents are not supported. Open the file and export it to PDF instead."
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CVInputError(
            f"Could not read the {label}. Provide it as PDF, markdown, or plain text."
        ) from exc


def _extract_pdf_text(raw: bytes, label: str) -> str:
    """Extract and join the text of every page of a PDF, raising CVInputError on error."""
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() for page in reader.pages]
    except PyPdfError as exc:
        raise CVInputError(f"Could not read the {label} PDF: {exc}") from exc
    return "\n".join(pages)

import io
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

MAX_DOCUMENT_BYTES = 5 * 1024 * 1024


class CVInputError(Exception):
    pass


def extract_cv_text(path: Path) -> str:
    return extract_document_text(path, label="CV")


def extract_document_text(path: Path, label: str) -> str:
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
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() for page in reader.pages]
    except PyPdfError as exc:
        raise CVInputError(f"Could not read the {label} PDF: {exc}") from exc
    return "\n".join(pages)

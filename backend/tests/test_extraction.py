from pathlib import Path

import pytest
from pdf_fixture import make_pdf

from sotellme.extraction import CVInputError, extract_cv_text


def test_plaintext_cv_is_returned_verbatim(tmp_path: Path) -> None:
    cv = tmp_path / "cv.txt"
    cv.write_text("Jane Doe\nSenior Engineer at Acme")

    assert extract_cv_text(cv) == "Jane Doe\nSenior Engineer at Acme"


def test_pdf_cv_text_is_extracted(tmp_path: Path) -> None:
    cv = tmp_path / "cv.pdf"
    cv.write_bytes(make_pdf(["Jane Doe", "Senior Engineer at Acme"]))

    text = extract_cv_text(cv)

    assert "Jane Doe" in text
    assert "Senior Engineer at Acme" in text
    assert "%PDF" not in text
    assert "endstream" not in text


def test_pdf_is_detected_by_content_not_extension(tmp_path: Path) -> None:
    cv = tmp_path / "cv.txt"
    cv.write_bytes(make_pdf(["Jane Doe"]))

    text = extract_cv_text(cv)

    assert "Jane Doe" in text
    assert "%PDF" not in text


def test_docx_cv_is_rejected_with_export_hint(tmp_path: Path) -> None:
    cv = tmp_path / "cv.docx"
    cv.write_bytes(b"PK\x03\x04" + b"\x00" * 64)

    with pytest.raises(CVInputError, match="export it to PDF"):
        extract_cv_text(cv)


def test_unreadable_binary_cv_is_rejected(tmp_path: Path) -> None:
    cv = tmp_path / "cv.txt"
    cv.write_bytes(b"\xff\xfe\x00\x01garbage")

    with pytest.raises(CVInputError, match="PDF, markdown, or plain text"):
        extract_cv_text(cv)


def test_cv_over_size_cap_is_rejected(tmp_path: Path) -> None:
    cv = tmp_path / "cv.txt"
    cv.write_bytes(b"x" * (5 * 1024 * 1024 + 1))

    with pytest.raises(CVInputError, match="5 MB"):
        extract_cv_text(cv)

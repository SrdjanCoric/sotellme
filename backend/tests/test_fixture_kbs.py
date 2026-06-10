from pathlib import Path

import sotellme


def test_package_importable() -> None:
    assert sotellme.__version__


def test_how_to_answer_fixture_kb_has_content(fixture_kb_how_to_answer: Path) -> None:
    docs = sorted(fixture_kb_how_to_answer.glob("*.md"))
    assert docs
    assert all(doc.read_text().strip() for doc in docs)


def test_how_to_interview_fixture_kb_has_content(fixture_kb_how_to_interview: Path) -> None:
    docs = sorted(fixture_kb_how_to_interview.glob("*.md"))
    assert docs
    assert all(doc.read_text().strip() for doc in docs)

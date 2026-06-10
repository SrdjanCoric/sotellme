from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_kb_how_to_answer() -> Path:
    return FIXTURES_DIR / "how-to-answer"


@pytest.fixture
def fixture_kb_how_to_interview() -> Path:
    return FIXTURES_DIR / "how-to-interview"

import json
import os
from pathlib import Path
from typing import Any

import pytest

from sotellme.config import PROVIDER_KEY_VARS, build_chat_model, resolve_model_config
from sotellme.coverage import StarFlags
from sotellme.flagger import flag_star_elements

CASES_FILE = Path(__file__).parent.parent / "evals" / "star_flagger_cases.json"

EVAL_PROVIDER = os.environ.get("SOTELLME_PROVIDER", "google_genai")

needs_provider_key = pytest.mark.skipif(
    not os.environ.get(PROVIDER_KEY_VARS.get(EVAL_PROVIDER, ""), ""),
    reason=f"star-flagger evals need a real {EVAL_PROVIDER} key",
)


def load_cases() -> list[dict[str, Any]]:
    document = json.loads(CASES_FILE.read_text())
    cases: list[dict[str, Any]] = document["cases"]
    return cases


@needs_provider_key
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: str(case["name"]))
def test_known_gap_answers_are_flagged_correctly(case: dict[str, Any]) -> None:
    config = resolve_model_config(env=os.environ, provider=EVAL_PROVIDER)
    model = build_chat_model(config, "fast")

    flags = flag_star_elements(case["answer"], model)

    assert flags == StarFlags(**case["expected"])

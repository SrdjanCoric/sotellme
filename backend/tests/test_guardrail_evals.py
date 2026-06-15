import json
import os
from pathlib import Path
from typing import Any

import pytest

from sotellme.config import PROVIDER_KEY_VARS, build_chat_model, resolve_model_config
from sotellme.guardrail import LLMGuardrail

CASES_FILE = Path(__file__).parent.parent / "evals" / "guardrail_cases.json"

EVAL_PROVIDER = os.environ.get("SOTELLME_PROVIDER", "google_genai")

needs_provider_key = pytest.mark.skipif(
    not os.environ.get(PROVIDER_KEY_VARS.get(EVAL_PROVIDER, ""), ""),
    reason=f"guardrail evals need a real {EVAL_PROVIDER} key",
)


def load_cases() -> list[dict[str, Any]]:
    document = json.loads(CASES_FILE.read_text())
    cases: list[dict[str, Any]] = document["cases"]
    return cases


@needs_provider_key
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: str(case["name"]))
def test_the_guardrail_screens_the_reply_correctly(case: dict[str, Any]) -> None:
    config = resolve_model_config(env=os.environ, provider=EVAL_PROVIDER)
    guardrail = LLMGuardrail(build_chat_model(config, "fast"))

    verdict = guardrail.classify(case["question"], case["answer"])

    assert verdict in case["allowed"], (
        f"guardrail screened {case['name']!r} as {verdict!r}; allowed: {case['allowed']}"
    )

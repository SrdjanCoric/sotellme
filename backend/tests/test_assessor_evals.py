import json
import os
from pathlib import Path
from typing import Any

import pytest

from sotellme.assessor import assess_answer
from sotellme.config import PROVIDER_KEY_VARS, build_chat_model, resolve_model_config
from sotellme.interviewer import Turn

CASES_FILE = Path(__file__).parent.parent / "evals" / "assessor_cases.json"

EVAL_PROVIDER = os.environ.get("SOTELLME_PROVIDER", "google_genai")

needs_provider_key = pytest.mark.skipif(
    not os.environ.get(PROVIDER_KEY_VARS.get(EVAL_PROVIDER, ""), ""),
    reason=f"assessor evals need a real {EVAL_PROVIDER} key",
)


def load_cases() -> list[dict[str, Any]]:
    document = json.loads(CASES_FILE.read_text())
    cases: list[dict[str, Any]] = document["cases"]
    return cases


@needs_provider_key
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: str(case["name"]))
def test_the_assessor_reads_the_answer_correctly(case: dict[str, Any]) -> None:
    config = resolve_model_config(env=os.environ, provider=EVAL_PROVIDER)
    model = build_chat_model(config, "fast")
    transcript = [Turn(question=f"Tell me about {case['topic']}.", answer=case["answer"])]

    assessment = assess_answer(case["topic"], transcript, model)

    expected = case["expected"]
    observed = {
        **assessment.star.model_dump(),
        "sufficient_signal": assessment.sufficient_signal,
    }
    mismatched = {flag: observed[flag] for flag in expected if observed[flag] != expected[flag]}
    assert not mismatched, f"assessor misread {case['name']}: {mismatched}"

    chased = " | ".join(assessment.claims_worth_chasing).lower()
    missing_claims = [
        substring
        for substring in case.get("claim_substrings", [])
        if substring.lower() not in chased
    ]
    assert not missing_claims, (
        f"claims worth chasing miss {missing_claims}; got {assessment.claims_worth_chasing}"
    )

import os
from typing import Any

import pytest
from test_grader_evals import (
    EVAL_PROVIDER,
    STAR_FLAGS,
    grade_case,
    load_cases,
)

from sotellme.assessor import assess_answer
from sotellme.config import PROVIDER_KEY_VARS, build_chat_model, resolve_model_config
from sotellme.interviewer import Turn

needs_provider_key = pytest.mark.skipif(
    not os.environ.get(PROVIDER_KEY_VARS.get(EVAL_PROVIDER, ""), ""),
    reason=f"assessor-vs-grader agreement needs a real {EVAL_PROVIDER} key",
)


def star_flags_of_assessor(case: dict[str, Any]) -> dict[str, bool]:
    config = resolve_model_config(env=os.environ, provider=EVAL_PROVIDER)
    model = build_chat_model(config, "fast")
    transcript = [Turn(question=case["question"], answer=case["answer"])]
    assessment = assess_answer(case["question"], transcript, model)
    return assessment.star.model_dump()


def agreement(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int]:
    matched = sum(1 for flag in STAR_FLAGS if a[flag] == b[flag])
    return matched, len(STAR_FLAGS)


@needs_provider_key
def test_grader_is_canonical_and_agreement_is_reported() -> None:
    """Grader is canonical: it is the reference the assessor's mid-session read is measured
    against. Reports grader-vs-human, assessor-vs-human, and assessor-vs-grader STAR agreement."""
    cases = load_cases()
    grader_human = assessor_human = assessor_grader = 0
    total = 0
    for case in cases:
        human = {flag: case["proposed"]["star"][flag] for flag in STAR_FLAGS}
        grader = grade_case(case).star.model_dump()
        assessor = star_flags_of_assessor(case)

        gh, n = agreement(grader, human)
        ah, _ = agreement(assessor, human)
        ag, _ = agreement(assessor, grader)
        grader_human += gh
        assessor_human += ah
        assessor_grader += ag
        total += n

    report = (
        f"STAR flag agreement over {total} flags - "
        f"grader-vs-human: {grader_human / total:.0%}, "
        f"assessor-vs-human: {assessor_human / total:.0%}, "
        f"assessor-vs-grader: {assessor_grader / total:.0%}"
    )
    print(report)
    assert grader_human / total >= 0.85, report

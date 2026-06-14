import json
import os
from pathlib import Path
from typing import Any

import pytest

from sotellme.config import PROVIDER_KEY_VARS, build_chat_model, resolve_model_config
from sotellme.grader import AnswerScore, grade_session
from sotellme.interviewer import Turn

CASES_FILE = Path(__file__).parent.parent / "evals" / "grader_cases.json"

EVAL_PROVIDER = os.environ.get("SOTELLME_PROVIDER", "google_genai")

needs_provider_key = pytest.mark.skipif(
    not os.environ.get(PROVIDER_KEY_VARS.get(EVAL_PROVIDER, ""), ""),
    reason=f"grader evals need a real {EVAL_PROVIDER} key",
)

STAR_FLAGS = ("situation", "task", "action", "result", "quantified_result")


def load_cases() -> list[dict[str, Any]]:
    document = json.loads(CASES_FILE.read_text())
    cases: list[dict[str, Any]] = document["cases"]
    return cases


def grade_case(case: dict[str, Any]) -> AnswerScore:
    config = resolve_model_config(env=os.environ, provider=EVAL_PROVIDER)
    model = build_chat_model(config, "smart")
    transcript = [Turn(question=case["question"], answer=case["answer"])]
    grade = grade_session(transcript, case["target_level"], model)
    assert len(grade.scores) == 1, f"{case['name']}: expected one score, got {len(grade.scores)}"
    return grade.scores[0]


def disagreements(answer: AnswerScore, expected: dict[str, Any]) -> dict[str, Any]:
    observed = answer.star.model_dump()
    found: dict[str, Any] = {}
    star_misses = {
        flag: observed[flag] for flag in STAR_FLAGS if observed[flag] != expected["star"][flag]
    }
    if star_misses:
        found["star"] = star_misses
    required = set(expected["weak_or_missing"])
    present = {flag for flag in STAR_FLAGS if expected["star"][flag]}
    allowed = required | present
    got = set(answer.weak_or_missing)
    if not (required <= got <= allowed):
        found["weak_or_missing"] = {
            "got": sorted(got),
            "want_at_least": sorted(required),
            "may_also_flag": sorted(present - required),
        }
    if answer.specificity != expected["specificity"]:
        found["specificity"] = {"got": answer.specificity, "want": expected["specificity"]}
    if answer.ownership != expected["ownership"]:
        found["ownership"] = {"got": answer.ownership, "want": expected["ownership"]}
    if abs(answer.score - expected["score"]) > 1:
        found["score"] = {"got": answer.score, "want": expected["score"]}
    return found


@needs_provider_key
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: str(case["name"]))
def test_the_grader_scores_the_answer_as_labeled(case: dict[str, Any]) -> None:
    answer = grade_case(case)

    found = disagreements(answer, case["proposed"])
    assert not found, (
        f"grader disagrees with the labels on {case['name']}: {found}\n"
        f"  question: {case['question']}\n"
        f"  answer: {case['answer']}\n"
        f"  grader specificity={answer.specificity} ownership={answer.ownership} "
        f"score={answer.score}\n"
        f"  grader rationale: {answer.rationale}\n"
        f"  grader gap: {answer.gap}"
    )

import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field
from voice import voice_tells

from sotellme.coach import CoachReport, coach_session
from sotellme.config import PROVIDER_KEY_VARS, build_chat_model, resolve_model_config
from sotellme.grader import SessionGrade
from sotellme.interviewer import Turn

CASES_FILE = Path(__file__).parent.parent / "evals" / "coach_cases.json"

EVAL_PROVIDER = os.environ.get("SOTELLME_PROVIDER", "google_genai")
JUDGE_PROVIDER = os.environ.get("SOTELLME_EVAL_JUDGE_PROVIDER", "anthropic")
JUDGE_MODEL = os.environ.get("SOTELLME_EVAL_JUDGE_MODEL") or None


def _missing_key(provider: str) -> bool:
    return not os.environ.get(PROVIDER_KEY_VARS.get(provider, ""), "")


needs_provider_key = pytest.mark.skipif(
    _missing_key(EVAL_PROVIDER) or _missing_key(JUDGE_PROVIDER),
    reason=f"coach evals need real {EVAL_PROVIDER} and {JUDGE_PROVIDER} keys",
)


def load_cases() -> list[dict[str, Any]]:
    document: dict[str, Any] = json.loads(CASES_FILE.read_text())
    cases: list[dict[str, Any]] = document["cases"]
    return cases


class CoachJudgeVerdict(BaseModel):
    addresses_the_named_gap: bool = Field(
        description=(
            "True only if the advice tells the candidate to fix the specific gap named below, "
            "on their own story, not some other weakness."
        )
    )
    is_generic: bool = Field(
        description=(
            "True if the advice is generic filler that could be pasted onto any weak answer, "
            "such as 'be more specific', 'add more detail', or 'quantify your impact', with no "
            "concrete pointer to what to add to this story."
        )
    )


JUDGE_SYSTEM_PROMPT = (
    "You evaluate a single piece of interview coaching. You are given the candidate's answer, "
    "the gap that was found in it, and the coach's advice for fixing it. Judge whether the "
    "advice genuinely addresses that named gap on the candidate's own material, and whether it "
    "is generic filler. Judge strictly and return your verdict."
)

JUDGE_HUMAN_TEMPLATE = (
    "The candidate's answer:\n<answer>\n{answer}\n</answer>\n"
    "The gap found in it:\n<gap>\n{gap}\n</gap>\n"
    "The coach's advice:\n<advice>\n{advice}\n</advice>"
)


def coach_case(case: dict[str, Any]) -> CoachReport:
    config = resolve_model_config(env=os.environ, provider=EVAL_PROVIDER)
    model = build_chat_model(config, "smart")
    transcript = [Turn(question=t["question"], answer=t["answer"]) for t in case["transcript"]]
    grade = SessionGrade.model_validate(case["grade"])
    return coach_session(transcript, grade, case["target_level"], model)


def judge_advice(answer: str, gap: str, advice: str) -> CoachJudgeVerdict:
    config = resolve_model_config(env=os.environ, provider=JUDGE_PROVIDER, smart_model=JUDGE_MODEL)
    judge = build_chat_model(config, "smart").with_structured_output(CoachJudgeVerdict)
    verdict = judge.invoke(
        [
            ("system", JUDGE_SYSTEM_PROMPT),
            ("human", JUDGE_HUMAN_TEMPLATE.format(answer=answer, gap=gap, advice=advice)),
        ]
    )
    assert isinstance(verdict, CoachJudgeVerdict)
    return verdict


def coaching_prose(report: CoachReport) -> str:
    parts = [report.summary, report.study_plan]
    for advice in report.answer_advice:
        parts.extend([advice.diagnosis, advice.fix])
    for drill in report.drills:
        parts.extend([drill.focus, drill.exercise])
    return "\n".join(parts)


@needs_provider_key
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: str(case["name"]))
def test_the_coach_ties_its_fix_to_the_named_gap(case: dict[str, Any]) -> None:
    report = coach_case(case)

    assert report.answer_advice, f"{case['name']}: the coach gave no advice on a weak answer"
    advice = report.answer_advice[0]
    answer = case["transcript"][0]["answer"]

    verdict = judge_advice(answer, case["gap_summary"], f"{advice.diagnosis}\n{advice.fix}")
    assert verdict.addresses_the_named_gap, (
        f"{case['name']}: advice does not address the named gap.\n"
        f"  gap: {case['gap_summary']}\n  diagnosis: {advice.diagnosis}\n  fix: {advice.fix}"
    )
    assert not verdict.is_generic, (
        f"{case['name']}: advice is generic filler, not tied to the story.\n  fix: {advice.fix}"
    )


@needs_provider_key
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: str(case["name"]))
def test_the_coaching_prose_keeps_the_house_voice(case: dict[str, Any]) -> None:
    report = coach_case(case)

    prose = coaching_prose(report)
    assert not voice_tells(prose), f"{case['name']}: voice tells in coaching: {voice_tells(prose)}"

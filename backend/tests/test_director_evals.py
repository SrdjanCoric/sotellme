import json
import os
from pathlib import Path
from typing import Any

import pytest

from sotellme.assessor import TopicAssessment
from sotellme.config import PROVIDER_KEY_VARS, build_chat_model, resolve_model_config
from sotellme.director import DirectorSituation, LLMDirector
from sotellme.interviewer import Turn
from sotellme.profile import CandidateProfile
from sotellme.role import RoleContext

CASES_FILE = Path(__file__).parent.parent / "evals" / "director_cases.json"

EVAL_PROVIDER = os.environ.get("SOTELLME_PROVIDER", "google_genai")

needs_provider_key = pytest.mark.skipif(
    not os.environ.get(PROVIDER_KEY_VARS.get(EVAL_PROVIDER, ""), ""),
    reason=f"director evals need a real {EVAL_PROVIDER} key",
)


def load_document() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(CASES_FILE.read_text())
    return document


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = load_document()["cases"]
    return cases


def situation_from(document: dict[str, Any], case: dict[str, Any]) -> DirectorSituation:
    return DirectorSituation(
        profile=CandidateProfile.model_validate(document["profile"]),
        context=RoleContext.model_validate(document["role_context"]),
        emphasis=tuple(document["emphasis"]),
        brief=document["brief"],
        transcript=[Turn.model_validate(turn) for turn in case["transcript"]],
        assessments=[TopicAssessment.model_validate(entry) for entry in case["assessments"]],
        questions_asked=case["questions_asked"],
        question_cap=document["question_cap"],
    )


@needs_provider_key
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: str(case["name"]))
def test_the_director_makes_the_right_call(case: dict[str, Any]) -> None:
    config = resolve_model_config(env=os.environ, provider=EVAL_PROVIDER)
    director = LLMDirector(build_chat_model(config, "fast"))

    decision = director.decide(situation_from(load_document(), case))

    assert decision.action in case["allowed_actions"], (
        f"director chose {decision.action!r} (subject: {decision.subject!r}, "
        f"reason: {decision.reason!r}); allowed: {case['allowed_actions']}"
    )

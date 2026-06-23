import pytest
from pydantic import ValidationError
from stubs import StubChatModel

from sotellme.assessor import AnswerAssessment, StarFlags, TopicAssessment
from sotellme.director import (
    DirectorDecision,
    DirectorError,
    DirectorSituation,
    LLMDirector,
    render_assessments,
    render_role_details,
)
from sotellme.interviewer import Turn
from sotellme.profile import CandidateProfile, Role
from sotellme.role import CompetencyWeight, RoleContext

PROFILE = CandidateProfile(
    roles=[Role(title="Engineer", organization="Acme")],
    projects=[],
    quantified_claims=["Cut latency by 38%"],
    technologies=["Python"],
)

CONTEXT = RoleContext(
    company="Acme",
    role_title="Backend Engineer",
    competencies=[
        CompetencyWeight(name="ownership", weight=5),
        CompetencyWeight(name="conflict", weight=2),
    ],
    target_level="staff",
)

ASSESSMENT = AnswerAssessment(
    star=StarFlags(situation=True, task=True, action=True, result=False, quantified_result=False),
    sufficient_signal=False,
    claims_worth_chasing=["rewrote the scheduler alone"],
)

DECISION = DirectorDecision(
    action="follow_up", subject="rewrote the scheduler alone", reason="ownership signal"
)

REPROMPT_DECISION = DirectorDecision(
    action="reprompt",
    subject="the rollout they skipped past",
    reason="a fair question went unanswered",
)


def situation(
    transcript: list[Turn] | None = None,
    log: list[TopicAssessment] | None = None,
    questions_asked: int = 0,
) -> DirectorSituation:
    return DirectorSituation(
        profile=PROFILE,
        context=CONTEXT,
        emphasis=("strategic leadership", "developing others"),
        brief="Acme builds billing software for veterinary clinics.",
        transcript=transcript or [],
        assessments=log or [],
        questions_asked=questions_asked,
        question_cap=20,
    )


def test_the_director_returns_a_structured_decision() -> None:
    model = StubChatModel(structured_response=DECISION)

    decision = LLMDirector(model).decide(situation())

    assert decision == DECISION


def test_the_director_can_reprompt_a_deflected_question() -> None:
    model = StubChatModel(structured_response=REPROMPT_DECISION)

    decision = LLMDirector(model).decide(situation())

    assert decision == REPROMPT_DECISION
    assert decision.action == "reprompt"


def test_the_director_sees_the_whole_situation() -> None:
    model = StubChatModel(structured_response=DECISION)
    transcript = [Turn(question="Tell me about Acme.", answer="I rewrote the scheduler.")]
    log = [TopicAssessment(topic="the Acme work", assessment=ASSESSMENT)]

    LLMDirector(model).decide(situation(transcript, log, questions_asked=1))

    seen = " ".join(text for _, text in model.seen_inputs[0])
    assert "Cut latency by 38%" in seen
    assert "billing software for veterinary clinics" in seen
    assert "I rewrote the scheduler." in seen
    assert "rewrote the scheduler alone" in seen
    assert "strategic leadership" in seen
    assert "1" in seen and "20" in seen


def test_the_director_never_sees_the_target_level() -> None:
    model = StubChatModel(structured_response=DECISION)

    LLMDirector(model).decide(situation())

    seen = " ".join(text for _, text in model.seen_inputs[0]).lower()
    assert "staff" not in seen
    assert "target level" not in seen


def test_a_failed_parse_is_a_clear_error() -> None:
    error = ValidationError.from_exception_data("DirectorDecision", [])
    model = StubChatModel(structured_error=error)

    with pytest.raises(DirectorError, match="Could not decide"):
        LLMDirector(model).decide(situation())


def test_render_role_details_carries_weights_and_never_the_level() -> None:
    text = render_role_details(CONTEXT)

    assert "Company: Acme" in text
    assert "ownership (5)" in text
    assert "conflict (2)" in text
    assert "staff" not in text.lower()


def test_render_assessments_reports_signal_claims_and_missing_elements() -> None:
    log = [TopicAssessment(topic="the Acme work", assessment=ASSESSMENT)]

    text = render_assessments(log)

    assert "the Acme work" in text
    assert "more signal" in text
    assert "rewrote the scheduler alone" in text
    assert "result" in text


def test_render_assessments_is_empty_safe() -> None:
    assert render_assessments([]) == "No answers assessed yet."

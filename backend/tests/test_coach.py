import pytest
from langchain_core.exceptions import OutputParserException
from stubs import StubChatModel

from sotellme.assessor import StarFlags
from sotellme.coach import AnswerAdvice, CoachingError, CoachReport, Drill, coach_session
from sotellme.grader import AnswerScore, SessionGrade
from sotellme.interviewer import Turn


def a_report() -> CoachReport:
    return CoachReport(
        summary="You tell a clear story but keep stopping short of the outcome.",
        answer_advice=[
            AnswerAdvice(
                question="Tell me about the migration.",
                diagnosis="You named the work but never said how it turned out.",
                fix="Close the migration story with the latency you measured after the cutover.",
            )
        ],
        drills=[
            Drill(
                focus="Stating the result",
                exercise="Retell your last project in four sentences, the last one a number.",
            )
        ],
        study_plan="Spend a session turning each project into a STAR story that ends on a metric.",
    )


def a_grade() -> SessionGrade:
    return SessionGrade(
        scores=[
            AnswerScore(
                question="Tell me about the migration.",
                turn_index=1,
                star=StarFlags(
                    situation=True, task=True, action=True, result=False, quantified_result=False
                ),
                specificity="medium",
                ownership="clear",
                weak_or_missing=["result", "quantified_result"],
                gap="The story never says how the migration turned out.",
                rationale="Clear ownership, but the outcome is missing.",
                score=3,
            )
        ]
    )


def test_coach_session_returns_the_models_structured_report() -> None:
    transcript = [Turn(question="Tell me about the migration.", answer="I led the cutover.")]
    report = a_report()
    model = StubChatModel(structured_response=report)

    result = coach_session(transcript, a_grade(), "senior", model)

    assert result == report


def test_coach_session_retries_a_transient_structured_output_failure() -> None:
    transcript = [Turn(question="Tell me about the migration.", answer="I led the cutover.")]
    report = a_report()
    model = StubChatModel(
        structured_response=report,
        structured_error=OutputParserException("dropped a field"),
        structured_error_limit=1,
    )

    result = coach_session(transcript, a_grade(), "senior", model)

    assert result == report
    assert model.structured_calls == 2


def test_coach_session_raises_coaching_error_when_the_output_does_not_validate() -> None:
    transcript = [Turn(question="Tell me about the migration.", answer="We did stuff.")]
    model = StubChatModel(structured_error=OutputParserException("malformed"))

    with pytest.raises(CoachingError):
        coach_session(transcript, a_grade(), "senior", model)


def test_coach_session_skips_the_model_when_there_are_no_scores() -> None:
    model = StubChatModel(structured_response=a_report())

    result = coach_session([], SessionGrade(scores=[]), "senior", model)

    assert result.answer_advice == []
    assert result.drills == []
    assert model.structured_calls == 0

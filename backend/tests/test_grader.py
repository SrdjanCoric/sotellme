import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError
from stubs import StubChatModel

from sotellme.assessor import StarFlags
from sotellme.eval_datasets import disagreements
from sotellme.grader import AnswerScore, GradingError, SessionGrade, grade_session
from sotellme.interviewer import Turn

_STAR_NAMES = ("situation", "task", "action", "result", "quantified_result")


def _expected(star: tuple[bool, ...], weak: list[str]) -> dict[str, object]:
    return {
        "star": dict(zip(_STAR_NAMES, star, strict=True)),
        "specificity": "high",
        "ownership": "clear",
        "weak_or_missing": weak,
        "score": 4,
    }


def _graded(star: tuple[bool, ...], weak: list[str]) -> AnswerScore:
    return AnswerScore(
        question="q",
        star=StarFlags(**dict(zip(_STAR_NAMES, star, strict=True))),
        specificity="high",
        ownership="clear",
        weak_or_missing=weak,  # type: ignore[arg-type]
        gap="",
        rationale="r",
        score=4,
    )


def complete_score(question: str) -> AnswerScore:
    return AnswerScore(
        question=question,
        star=StarFlags(situation=True, task=True, action=True, result=True, quantified_result=True),
        specificity="high",
        ownership="clear",
        weak_or_missing=[],
        gap="",
        rationale="Complete, quantified, single-team story that lands at the target level.",
        score=5,
    )


def test_disagreements_tolerates_a_weak_flag_on_a_present_element() -> None:
    expected = _expected((True, True, True, True, False), ["quantified_result"])
    graded = _graded((True, True, True, True, False), ["action", "quantified_result"])

    assert "weak_or_missing" not in disagreements(graded, expected)


def test_disagreements_rejects_flagging_an_absent_element_the_label_omitted() -> None:
    expected = _expected((False, False, True, False, False), [])
    graded = _graded((False, False, True, False, False), ["situation", "task"])

    assert "weak_or_missing" in disagreements(graded, expected)


def test_disagreements_requires_every_labeled_weak_element() -> None:
    expected = _expected((True, True, True, True, False), ["quantified_result"])
    graded = _graded((True, True, True, True, False), [])

    assert "weak_or_missing" in disagreements(graded, expected)


def test_answer_score_accepts_not_applicable_ownership() -> None:
    answer = AnswerScore(
        question="Why do you want to work here?",
        star=StarFlags(
            situation=False, task=False, action=False, result=False, quantified_result=False
        ),
        specificity="high",
        ownership="not_applicable",
        weak_or_missing=[],
        gap="",
        rationale="Motivation answer; ownership does not apply with no personal action claimed.",
        score=4,
    )

    assert answer.ownership == "not_applicable"


def test_answer_score_requires_a_rationale() -> None:
    with pytest.raises(ValidationError):
        AnswerScore(  # type: ignore[call-arg]
            question="Tell me about the migration.",
            star=StarFlags(
                situation=False, task=False, action=False, result=False, quantified_result=False
            ),
            specificity="low",
            ownership="unclear",
            weak_or_missing=[],
            gap="",
            score=1,
        )


def test_grade_session_returns_the_models_structured_grade() -> None:
    transcript = [
        Turn(question="Tell me about the migration.", answer="I led it; latency fell 40%.")
    ]
    grade = SessionGrade(scores=[complete_score(transcript[0].question)])
    model = StubChatModel(structured_response=grade)

    result = grade_session(transcript, "senior", model)

    assert result == grade


def test_grade_session_retries_a_transient_structured_output_failure() -> None:
    transcript = [
        Turn(question="Tell me about the migration.", answer="I led it; latency fell 40%.")
    ]
    grade = SessionGrade(scores=[complete_score(transcript[0].question)])
    model = StubChatModel(
        structured_response=grade,
        structured_error=OutputParserException("dropped the score field"),
        structured_error_limit=1,
    )

    result = grade_session(transcript, "senior", model)

    assert result == grade
    assert model.structured_calls == 2


def test_grade_session_raises_grading_error_when_the_output_does_not_validate() -> None:
    transcript = [Turn(question="Tell me about the migration.", answer="We did stuff.")]
    model = StubChatModel(structured_error=OutputParserException("malformed"))

    with pytest.raises(GradingError):
        grade_session(transcript, "senior", model)

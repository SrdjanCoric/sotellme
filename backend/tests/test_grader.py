import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError
from stubs import StubChatModel

from sotellme.assessor import StarFlags
from sotellme.eval_datasets import ExpectedAnswer, disagreements
from sotellme.grader import (
    AnswerScore,
    GradingError,
    SessionGrade,
    SkippedTurn,
    grade_session,
)
from sotellme.interviewer import Turn

_STAR_NAMES = ("situation", "task", "action", "result", "quantified_result")


def _expected(star: tuple[bool, ...], weak: list[str]) -> ExpectedAnswer:
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
        turn_index=1,
        star=StarFlags(**dict(zip(_STAR_NAMES, star, strict=True))),
        specificity="high",
        ownership="clear",
        weak_or_missing=weak,  # type: ignore[arg-type]
        gap="One refinement short of a five.",
        rationale="r",
        score=4,
    )


def complete_score(question: str, turn_index: int = 1) -> AnswerScore:
    return AnswerScore(
        question=question,
        turn_index=turn_index,
        star=StarFlags(situation=True, task=True, action=True, result=True, quantified_result=True),
        specificity="high",
        ownership="clear",
        weak_or_missing=[],
        gap="",
        rationale="Complete, quantified, single-team story that lands at the target level.",
        score=5,
    )


def test_session_grade_records_skipped_clarifying_turns() -> None:
    grade = SessionGrade(
        scores=[complete_score("Tell me about the migration.")],
        skipped=[
            SkippedTurn(
                turn_index=2,
                question="Did you mean the first or second migration?",
                reason="Clarifying question; no STAR substance to score.",
            )
        ],
    )

    assert grade.skipped[0].turn_index == 2


def test_session_grade_skipped_defaults_to_empty() -> None:
    grade = SessionGrade(scores=[complete_score("Tell me about the migration.")])

    assert grade.skipped == []


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


def test_answer_score_carries_the_transcript_turn_it_grades() -> None:
    answer = AnswerScore(
        question="Tell me about the migration.",
        turn_index=3,
        star=StarFlags(situation=True, task=True, action=True, result=True, quantified_result=True),
        specificity="high",
        ownership="clear",
        weak_or_missing=[],
        gap="",
        rationale="A complete, quantified story at the target level.",
        score=5,
    )

    assert answer.turn_index == 3


def test_answer_score_accepts_not_applicable_ownership() -> None:
    answer = AnswerScore(
        question="Why do you want to work here?",
        turn_index=1,
        star=StarFlags(
            situation=False, task=False, action=False, result=False, quantified_result=False
        ),
        specificity="high",
        ownership="not_applicable",
        weak_or_missing=[],
        gap="Strong motivation, but it could name one concrete draw to the product.",
        rationale="Motivation answer; ownership does not apply with no personal action claimed.",
        score=4,
    )

    assert answer.ownership == "not_applicable"


def test_answer_score_requires_a_rationale() -> None:
    with pytest.raises(ValidationError):
        AnswerScore(  # type: ignore[call-arg]
            question="Tell me about the migration.",
            turn_index=1,
            star=StarFlags(
                situation=False, task=False, action=False, result=False, quantified_result=False
            ),
            specificity="low",
            ownership="unclear",
            weak_or_missing=[],
            gap="No concrete substance to assess.",
            score=1,
        )


def _scored(score: int, gap: str) -> AnswerScore:
    return AnswerScore(
        question="Tell me about the migration.",
        turn_index=1,
        star=StarFlags(situation=True, task=True, action=True, result=True, quantified_result=True),
        specificity="high",
        ownership="clear",
        weak_or_missing=[],
        gap=gap,
        rationale="A complete, quantified story, judged at the target level.",
        score=score,
    )


def test_answer_score_rejects_an_empty_gap_below_a_five() -> None:
    with pytest.raises(ValidationError):
        _scored(4, "")


def test_answer_score_rejects_a_gap_on_a_five() -> None:
    with pytest.raises(ValidationError):
        _scored(5, "Single-team scope, not the cross-org reach a 5 would show.")


def test_answer_score_accepts_a_gap_below_a_five() -> None:
    answer = _scored(4, "Single-team scope, not the cross-org reach a 5 would show.")

    assert answer.score == 4
    assert answer.gap


def test_answer_score_accepts_an_empty_gap_on_a_five() -> None:
    answer = _scored(5, "")

    assert answer.score == 5
    assert answer.gap == ""


def test_answer_score_treats_a_whitespace_only_gap_as_empty() -> None:
    with pytest.raises(ValidationError):
        _scored(4, "   ")


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


def test_grade_session_accepts_a_grade_that_records_a_skipped_turn() -> None:
    transcript = [
        Turn(question="Tell me about the migration.", answer="I led it; latency fell 40%."),
        Turn(question="Did you mean the first or the second?", answer="The second one."),
    ]
    grade = SessionGrade(
        scores=[complete_score(transcript[0].question, turn_index=1)],
        skipped=[
            SkippedTurn(
                turn_index=2,
                question=transcript[1].question,
                reason="Clarifying question; no STAR substance to score.",
            )
        ],
    )
    model = StubChatModel(structured_response=grade)

    result = grade_session(transcript, "senior", model)

    assert result == grade


def test_grade_session_rejects_a_grade_that_leaves_a_turn_unaccounted() -> None:
    transcript = [
        Turn(question="Tell me about the migration.", answer="I led it; latency fell 40%."),
        Turn(question="Did you mean the first or the second?", answer="The second one."),
    ]
    grade = SessionGrade(scores=[complete_score(transcript[0].question, turn_index=1)])
    model = StubChatModel(structured_response=grade)

    with pytest.raises(GradingError):
        grade_session(transcript, "senior", model)


def test_grade_session_raises_grading_error_when_the_output_does_not_validate() -> None:
    transcript = [Turn(question="Tell me about the migration.", answer="We did stuff.")]
    model = StubChatModel(structured_error=OutputParserException("malformed"))

    with pytest.raises(GradingError):
        grade_session(transcript, "senior", model)


def test_grade_session_error_does_not_blame_a_short_transcript_for_a_coverage_failure() -> None:
    transcript = [
        Turn(question=f"Question {i}", answer="A complete, quantified STAR story.")
        for i in range(1, 8)
    ]
    grade = SessionGrade(scores=[complete_score(transcript[0].question, turn_index=1)])
    model = StubChatModel(structured_response=grade)

    with pytest.raises(GradingError) as exc_info:
        grade_session(transcript, "senior", model)

    assert "too short" not in str(exc_info.value).lower()
    detail = exc_info.value.diagnostic()
    assert "cover turns 1..7" in detail
    assert "OutputParserException" in detail


def test_a_grading_error_diagnostic_surfaces_the_underlying_cause() -> None:
    try:
        SessionGrade.model_validate({"scores": "not a list"})
    except ValidationError as cause:
        exc = GradingError("Could not grade the session.")
        exc.__cause__ = cause

    detail = exc.diagnostic()

    assert "Could not grade the session." in detail
    assert "ValidationError" in detail


def test_a_grading_error_diagnostic_without_a_cause_returns_the_message() -> None:
    assert GradingError("boom").diagnostic() == "boom"

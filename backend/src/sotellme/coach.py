"""Turn a graded session into candidate-facing coaching advice and drills."""

from collections.abc import Sequence

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

from sotellme.caching import cache_system_prompt
from sotellme.grader import SessionGrade
from sotellme.interviewer import Turn, render_transcript
from sotellme.prompts import coach_messages
from sotellme.role import TargetLevel


class AnswerAdvice(BaseModel):
    """Targeted coaching for one answer: what held it back and how to fix it."""

    question: str = Field(
        description="The interviewer question this advice is about, quoted near-verbatim."
    )
    diagnosis: str = Field(
        description=(
            "What specifically held this answer back, named against what the candidate "
            "actually said, not in the abstract."
        )
    )
    fix: str = Field(
        description=(
            "The concrete thing to do differently next time, tied to this answer's own gap: "
            "what to add, name, or quantify. Never generic advice like 'be more specific'."
        )
    )


class Drill(BaseModel):
    """A practice exercise that builds a recurring weak area."""

    focus: str = Field(
        description="The recurring weakness this drill builds, named in plain words."
    )
    exercise: str = Field(
        description="A concrete practice exercise the candidate can run on their own to build it."
    )


class CoachReport(BaseModel):
    """Full coaching report for a session: summary, per-answer advice, drills, and plan."""

    summary: str = Field(
        description=(
            "A candid read of how the session went across the answers, in the house voice: "
            "what is already working and what most needs work."
        )
    )
    answer_advice: list[AnswerAdvice] = Field(
        description=(
            "One entry per answer that needs work, in transcript order. Strong answers are "
            "left out."
        )
    )
    drills: list[Drill] = Field(
        description=(
            "A drill per recurring weak area the session surfaced. Empty when nothing recurs."
        )
    )
    study_plan: str = Field(
        description=(
            "A short plan aggregating the weak areas into what to work on, in priority order."
        )
    )


class CoachingError(Exception):
    """Raised when the session cannot be coached into a valid CoachReport."""

    pass


_COACH_FAILURE_MESSAGE = "Could not coach the session. The interview may be too short to coach."

_COACH_RETRY_ATTEMPTS = 3


def render_grade(grade: SessionGrade) -> str:
    """Render a session grade into a plain-text block for the coach prompt."""
    blocks: list[str] = []
    for index, answer in enumerate(grade.scores, start=1):
        lines = [
            f"Answer {index} (scored {answer.score}/5)",
            f"  question: {answer.question}",
            f"  specificity: {answer.specificity}; ownership: {answer.ownership}",
        ]
        if answer.weak_or_missing:
            lines.append(f"  weak or missing: {', '.join(answer.weak_or_missing)}")
        if answer.gap:
            lines.append(f"  gap: {answer.gap}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def coach_session(
    transcript: Sequence[Turn],
    grade: SessionGrade,
    target_level: TargetLevel,
    model: BaseChatModel,
    provider: str = "",
) -> CoachReport:
    """Produce a coaching report for a graded session via the model.

    Returns an empty report when the grade has no scores. Otherwise renders the
    transcript and grade into the coach prompt, caches the system prompt for the
    provider, and invokes the model with structured output, retrying on validation or
    parsing failures.

    Args:
        transcript: The interview turns the grade was produced from.
        grade: The session grade to coach against.
        target_level: The seniority level the coaching targets.
        model: The chat model used to produce the structured report.
        provider: The model provider name used to select prompt caching behavior.

    Returns:
        A CoachReport, empty when the grade has no scores.

    Raises:
        CoachingError: If the model output fails validation or parsing across all
            retries, or is not a CoachReport.
    """
    if not grade.scores:
        return CoachReport(summary="", answer_advice=[], drills=[], study_plan="")
    structured = model.with_structured_output(CoachReport).with_retry(
        retry_if_exception_type=(ValidationError, OutputParserException),
        wait_exponential_jitter=False,
        stop_after_attempt=_COACH_RETRY_ATTEMPTS,
    )
    messages = cache_system_prompt(
        coach_messages(target_level, render_transcript(transcript), render_grade(grade)), provider
    )
    try:
        result = structured.invoke(messages)
    except (ValidationError, OutputParserException) as exc:
        raise CoachingError(_COACH_FAILURE_MESSAGE) from exc
    if not isinstance(result, CoachReport):
        raise CoachingError(_COACH_FAILURE_MESSAGE)
    return result

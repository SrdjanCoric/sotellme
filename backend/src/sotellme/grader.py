"""Grade an interview transcript answer by answer at the candidate's target level."""

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field, ValidationError, model_validator

from sotellme.assessor import StarFlags
from sotellme.caching import cache_system_prompt
from sotellme.interviewer import Turn, render_transcript
from sotellme.prompts import grader_messages
from sotellme.role import TargetLevel

StarElement = Literal["situation", "task", "action", "result", "quantified_result"]


class AnswerScore(BaseModel):
    """Structured score for a single candidate answer at the target level."""

    question: str = Field(
        description="The interviewer question this answer responds to, quoted near-verbatim."
    )
    turn_index: int = Field(
        ge=1,
        description="The 1-based transcript turn ('Turn N') this answer is scored from.",
    )
    rationale: str = Field(
        description=(
            "Work through the answer here first, in one or two plain sentences: why it earns "
            "the score, specificity, and ownership reads that follow, naming the evidence "
            "behind each. Internal review only, never shown to the candidate."
        )
    )
    star: StarFlags = Field(description="Which STAR story elements the answer actually states.")
    specificity: Literal["low", "medium", "high"] = Field(
        description=(
            "How much of the answer is concrete rather than vague. 'high' is concrete "
            "throughout (named systems, numbers, specific decisions); 'low' leans on "
            "vague words with nothing concrete behind them; 'medium' names at least one "
            "concrete detail but leans vague for the rest."
        )
    )
    ownership: Literal["clear", "mixed", "unclear", "not_applicable"] = Field(
        description=(
            "How clearly the answer separates what the candidate personally did ('I') "
            "from what the team did ('we'). Reads only the I-vs-we line, never how strong "
            "or specific the answer is. 'not_applicable' when the answer claims no personal "
            "action at all, such as a motivation answer."
        )
    )
    weak_or_missing: list[StarElement] = Field(
        description=(
            "The STAR elements the answer leaves weak or absent, named. "
            "Empty when the answer is STAR-complete."
        )
    )
    gap: str = Field(
        description=(
            "One plain sentence on what most holds this answer back, for the coach to "
            "act on. Empty only on a 5; non-empty for every score below 5."
        )
    )
    score: int = Field(
        ge=1,
        le=5,
        description="Overall strength of the answer at the target level, 1 (weak) to 5 (strong).",
    )

    @model_validator(mode="after")
    def _gap_is_empty_only_on_a_five(self) -> "AnswerScore":
        """Enforce the visible-deduction rule: a sub-5 names its gap, a 5 has none.

        Keeps the grader honest about silent deductions - every score below 5 must say
        what holds the answer back, and a flag-free 5 carries no gap.
        """
        has_gap = bool(self.gap.strip())
        if self.score == 5 and has_gap:
            raise ValueError("a 5 is flag-free, so its gap must be empty")
        if self.score < 5 and not has_gap:
            raise ValueError("a score below 5 must name the gap that holds the answer back")
        return self


class SkippedTurn(BaseModel):
    """A transcript turn left unscored because it was not a STAR-gradeable answer."""

    turn_index: int = Field(
        ge=1, description="The 1-based transcript turn ('Turn N') that was not scored."
    )
    question: str = Field(
        description="The interviewer question for this turn, quoted near-verbatim."
    )
    reason: str = Field(
        description=(
            "One plain phrase saying why this turn was not scored, such as a clarifying or "
            "confirmation question with no STAR substance to grade."
        )
    )


class SessionGrade(BaseModel):
    """Grades for every scored answer in a session, plus the turns left unscored."""

    scores: list[AnswerScore] = Field(
        description="One score per substantive answer the candidate gave, in transcript order."
    )
    skipped: list[SkippedTurn] = Field(
        default_factory=list,
        description=(
            "Turns left unscored because they were not STAR-gradeable answers - a clarifying "
            "or confirmation question the candidate simply answered. Every transcript turn "
            "appears exactly once across scores and skipped."
        ),
    )


class GradingError(Exception):
    """Raised when the session cannot be graded into a valid SessionGrade."""

    def diagnostic(self) -> str:
        """Describe this failure with its chained cause so a run is diagnosable.

        The bare message is generic, and a wrapper that reduces a raised exception to its
        message (e.g. Langfuse's ``run_experiment``) hides the real
        ``ValidationError``/``OutputParserException``. Folding the chained ``__cause__``
        into the message keeps the failure diagnosable from the run output alone.
        """
        cause = self.__cause__
        if cause is None:
            return str(self)
        return f"{self} (caused by {type(cause).__name__}: {cause})"


_GRADE_FAILURE_MESSAGE = "Could not grade the session."


_GRADE_RETRY_ATTEMPTS = 3


def _require_full_turn_coverage(grade: SessionGrade, turn_count: int) -> SessionGrade:
    """Verify every transcript turn is scored or skipped exactly once.

    Raised as an OutputParserException so the structured-output retry re-runs the grader
    rather than letting a grade that drops or double-counts a turn slip through.
    """
    covered = sorted(
        [score.turn_index for score in grade.scores] + [skip.turn_index for skip in grade.skipped]
    )
    if covered != list(range(1, turn_count + 1)):
        raise OutputParserException(
            f"grade must cover turns 1..{turn_count} exactly once, got {covered}"
        )
    return grade


def grade_session(
    transcript: Sequence[Turn],
    target_level: TargetLevel,
    model: BaseChatModel,
    provider: str = "",
) -> SessionGrade:
    """Score every answer in the transcript at the target level via the model.

    Renders the transcript into the grader prompt, caches the system prompt for the
    provider, and invokes the model with structured output, retrying on validation or
    parsing failures.

    Args:
        transcript: The interview turns to grade, in order.
        target_level: The seniority level to grade the answers against.
        model: The chat model used to produce the structured grade.
        provider: The model provider name used to select prompt caching behavior.

    Returns:
        A SessionGrade holding one score per answer.

    Raises:
        GradingError: If the model output fails validation or parsing across all
            retries, or is not a SessionGrade.
    """

    def _check_coverage(grade: dict[str, Any] | BaseModel) -> SessionGrade:
        if not isinstance(grade, SessionGrade):
            raise GradingError(_GRADE_FAILURE_MESSAGE)
        return _require_full_turn_coverage(grade, len(transcript))

    graded: Runnable[Any, SessionGrade] = model.with_structured_output(
        SessionGrade
    ) | RunnableLambda(_check_coverage)
    structured = graded.with_retry(
        retry_if_exception_type=(ValidationError, OutputParserException),
        wait_exponential_jitter=False,
        stop_after_attempt=_GRADE_RETRY_ATTEMPTS,
    )
    messages = cache_system_prompt(
        grader_messages(target_level, render_transcript(transcript, numbered=True)), provider
    )
    try:
        result = structured.invoke(messages)
    except (ValidationError, OutputParserException) as exc:
        raise GradingError(_GRADE_FAILURE_MESSAGE) from exc
    if not isinstance(result, SessionGrade):
        raise GradingError(_GRADE_FAILURE_MESSAGE)
    return result

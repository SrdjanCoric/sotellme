from collections.abc import Sequence
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

from sotellme.assessor import StarFlags
from sotellme.interviewer import Turn, render_transcript
from sotellme.prompts import grader_messages
from sotellme.role import TargetLevel

StarElement = Literal["situation", "task", "action", "result", "quantified_result"]


class AnswerScore(BaseModel):
    question: str = Field(
        description="The interviewer question this answer responds to, quoted near-verbatim."
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
            "action at all, such as a motivation answer or a short clarifying reply."
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
            "act on. Empty when the answer is strong."
        )
    )
    score: int = Field(
        ge=1,
        le=5,
        description="Overall strength of the answer at the target level, 1 (weak) to 5 (strong).",
    )


class SessionGrade(BaseModel):
    scores: list[AnswerScore] = Field(
        description="One score per answer the candidate gave, in transcript order."
    )


class GradingError(Exception):
    pass


_GRADE_FAILURE_MESSAGE = "Could not grade the session. The interview may be too short to score."


_GRADE_RETRY_ATTEMPTS = 3


def grade_session(
    transcript: Sequence[Turn], target_level: TargetLevel, model: BaseChatModel
) -> SessionGrade:
    structured = model.with_structured_output(SessionGrade).with_retry(
        retry_if_exception_type=(ValidationError, OutputParserException),
        wait_exponential_jitter=False,
        stop_after_attempt=_GRADE_RETRY_ATTEMPTS,
    )
    try:
        result = structured.invoke(grader_messages(target_level, render_transcript(transcript)))
    except (ValidationError, OutputParserException) as exc:
        raise GradingError(_GRADE_FAILURE_MESSAGE) from exc
    if not isinstance(result, SessionGrade):
        raise GradingError(_GRADE_FAILURE_MESSAGE)
    return result

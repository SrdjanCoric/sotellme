from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

from sotellme.caching import cache_system_prompt
from sotellme.interviewer import Turn, render_transcript
from sotellme.prompts import coverage_judge_messages, question_judge_messages
from sotellme.role import TargetLevel, level_emphasis

Verdict = Literal["good", "weak", "bad"]


class DimensionVerdict(BaseModel):
    rationale: str = Field(description="Why this score, written before the score is chosen.")
    score: int = Field(ge=1, le=5, description="1 (poor) to 5 (excellent) on this dimension.")


class QuestionVerdict(BaseModel):
    relevance: DimensionVerdict = Field(
        description="Does the question target the competency in play?"
    )
    probes_the_flagged_gap: DimensionVerdict = Field(
        description="Does it chase the specific flagged gap rather than a generic question?"
    )
    level_appropriateness: DimensionVerdict = Field(
        description="Is the depth right for the target level (neither over- nor under-shooting)?"
    )
    non_leading: DimensionVerdict = Field(
        description="Does it avoid handing over the answer or presupposing a conclusion?"
    )
    follow_up_discipline: DimensionVerdict = Field(
        description="Was probing again vs moving on the right call given the evidence at the time?"
    )
    overall_rationale: str = Field(
        description="Why the overall verdict, drawn from the dimensions."
    )
    overall: Verdict = Field(description="Overall question quality: good, weak, or bad.")


class CompetencyCoverage(BaseModel):
    competency: str
    status: Literal["covered", "partially", "missed"]


class CoverageVerdict(BaseModel):
    competencies: list[CompetencyCoverage] = Field(
        description="Per-competency coverage status across the whole session."
    )
    rationale: str = Field(description="Why this coverage verdict.")
    verdict: Verdict = Field(description="Overall session-coverage verdict: good, weak, or bad.")


@dataclass
class QuestionContext:
    question: str
    competency: str
    target_level: TargetLevel
    gap: str
    transcript: Sequence[Turn] = field(default_factory=list)
    sufficient_signal: bool = False
    consecutive_follow_ups: int = 0


class JudgeError(Exception):
    pass


_FAILURE_MESSAGE = "Could not judge the question."


class QuestionJudge:
    def __init__(self, model: BaseChatModel, provider: str = "") -> None:
        self._model = model
        self._provider = provider

    def judge_question(self, context: QuestionContext) -> QuestionVerdict:
        structured = self._model.with_structured_output(QuestionVerdict)
        try:
            messages = cache_system_prompt(
                question_judge_messages(
                    question=context.question,
                    competency=context.competency,
                    target_level=context.target_level,
                    gap=context.gap,
                    transcript_text=render_transcript(context.transcript),
                    sufficient_signal=context.sufficient_signal,
                    consecutive_follow_ups=context.consecutive_follow_ups,
                ),
                self._provider,
            )
            result = structured.invoke(messages)
        except (ValidationError, OutputParserException) as exc:
            raise JudgeError(_FAILURE_MESSAGE) from exc
        if not isinstance(result, QuestionVerdict):
            raise JudgeError(_FAILURE_MESSAGE)
        return result

    def judge_coverage(
        self, target_level: TargetLevel, transcript: Sequence[Turn]
    ) -> CoverageVerdict:
        structured = self._model.with_structured_output(CoverageVerdict)
        try:
            messages = cache_system_prompt(
                coverage_judge_messages(
                    target_level=target_level,
                    emphasis=", ".join(level_emphasis(target_level)),
                    transcript_text=render_transcript(transcript),
                ),
                self._provider,
            )
            result = structured.invoke(messages)
        except (ValidationError, OutputParserException) as exc:
            raise JudgeError(_FAILURE_MESSAGE) from exc
        if not isinstance(result, CoverageVerdict):
            raise JudgeError(_FAILURE_MESSAGE)
        return result

"""Judge interviewer question quality and session competency coverage via the model."""

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
    """A score with its rationale for one judged dimension."""

    rationale: str = Field(description="Why this score, written before the score is chosen.")
    score: int = Field(ge=1, le=5, description="1 (poor) to 5 (excellent) on this dimension.")


class QuestionVerdict(BaseModel):
    """Per-dimension and overall verdict on a single interviewer question."""

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

    @property
    def dimensions(self) -> dict[str, DimensionVerdict]:
        """Map each scored dimension name to its verdict."""
        return {
            "relevance": self.relevance,
            "probes_the_flagged_gap": self.probes_the_flagged_gap,
            "level_appropriateness": self.level_appropriateness,
            "non_leading": self.non_leading,
            "follow_up_discipline": self.follow_up_discipline,
        }


class CompetencyCoverage(BaseModel):
    """Coverage status for one competency across the session."""

    competency: str
    status: Literal["covered", "partially", "missed"]


class CoverageVerdict(BaseModel):
    """Verdict on how well the session covered the target competencies."""

    competencies: list[CompetencyCoverage] = Field(
        description="Per-competency coverage status across the whole session."
    )
    rationale: str = Field(description="Why this coverage verdict.")
    verdict: Verdict = Field(description="Overall session-coverage verdict: good, weak, or bad.")


@dataclass
class QuestionContext:
    """The context a single interviewer question is judged against."""

    question: str
    competency: str
    target_level: TargetLevel
    gap: str
    transcript: Sequence[Turn] = field(default_factory=list)
    sufficient_signal: bool = False
    consecutive_follow_ups: int = 0


class JudgeError(Exception):
    """Raised when a question or coverage judgment cannot be produced."""

    pass


_FAILURE_MESSAGE = "Could not judge the question."


class QuestionJudge:
    """Judge question quality and session coverage by prompting a chat model."""

    def __init__(self, model: BaseChatModel, provider: str = "") -> None:
        self._model = model
        self._provider = provider

    def judge_question(self, context: QuestionContext) -> QuestionVerdict:
        """Judge one interviewer question against its context.

        Renders the context into the question-judge prompt, caches the system prompt
        for the provider, and invokes the model with structured output.

        Args:
            context: The question and its surrounding context to judge.

        Returns:
            The per-dimension and overall verdict on the question.

        Raises:
            JudgeError: If the model output fails validation or parsing, or is not a
                QuestionVerdict.
        """
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
        """Judge how well a session covered the target level's competencies.

        Renders the target level, its emphasis, and the transcript into the
        coverage-judge prompt, caches the system prompt for the provider, and invokes
        the model with structured output.

        Args:
            target_level: The seniority level whose competencies are judged.
            transcript: The interview turns to assess coverage over.

        Returns:
            The per-competency and overall coverage verdict.

        Raises:
            JudgeError: If the model output fails validation or parsing, or is not a
                CoverageVerdict.
        """
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

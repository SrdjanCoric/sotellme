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
    """Per-dimension and overall verdict on a single interviewer question.

    Flat scalar fields keep the tool-call schema shallow so Anthropic structured output
    stays reliable; five identically-shaped nested sub-objects collapse in practice. Each
    dimension keeps its rationale before its score, and the ``dimensions`` property
    reassembles ``DimensionVerdict`` values so downstream consumers are untouched.
    """

    relevance_rationale: str = Field(
        description="Why the relevance score, written before the score is chosen."
    )
    relevance_score: int = Field(
        ge=1, le=5, description="Does the question target the competency in play? 1 (poor)-5."
    )
    probes_the_flagged_gap_rationale: str = Field(
        description="Why the probes-the-flagged-gap score, written before the score."
    )
    probes_the_flagged_gap_score: int = Field(
        ge=1,
        le=5,
        description="Does it chase the specific flagged gap rather than a generic question? 1-5.",
    )
    level_appropriateness_rationale: str = Field(
        description="Why the level-appropriateness score, written before the score."
    )
    level_appropriateness_score: int = Field(
        ge=1,
        le=5,
        description="Is the depth right for the target level (no over- or under-shoot)? 1-5.",
    )
    non_leading_rationale: str = Field(
        description="Why the non-leading score, written before the score."
    )
    non_leading_score: int = Field(
        ge=1,
        le=5,
        description="Does it avoid handing over the answer or presupposing a conclusion? 1-5.",
    )
    follow_up_discipline_rationale: str = Field(
        description="Why the follow-up-discipline score, written before the score."
    )
    follow_up_discipline_score: int = Field(
        ge=1,
        le=5,
        description="Was probing again vs moving on the right call given the evidence then? 1-5.",
    )
    overall_rationale: str = Field(
        description="Why the overall verdict, drawn from the dimensions."
    )
    overall: Verdict = Field(description="Overall question quality: good, weak, or bad.")

    @property
    def dimensions(self) -> dict[str, DimensionVerdict]:
        """Reassemble each scored dimension from its flat rationale/score pair."""
        return {
            "relevance": DimensionVerdict(
                rationale=self.relevance_rationale, score=self.relevance_score
            ),
            "probes_the_flagged_gap": DimensionVerdict(
                rationale=self.probes_the_flagged_gap_rationale,
                score=self.probes_the_flagged_gap_score,
            ),
            "level_appropriateness": DimensionVerdict(
                rationale=self.level_appropriateness_rationale,
                score=self.level_appropriateness_score,
            ),
            "non_leading": DimensionVerdict(
                rationale=self.non_leading_rationale, score=self.non_leading_score
            ),
            "follow_up_discipline": DimensionVerdict(
                rationale=self.follow_up_discipline_rationale,
                score=self.follow_up_discipline_score,
            ),
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


_FAILURE_MESSAGE = "Could not judge the question."

_JUDGE_RETRY_ATTEMPTS = 3


class QuestionJudge:
    """Judge question quality and session coverage by prompting a chat model."""

    def __init__(self, model: BaseChatModel, provider: str = "") -> None:
        self._model = model
        self._provider = provider

    def judge_question(self, context: QuestionContext) -> QuestionVerdict:
        """Judge one interviewer question against its context.

        Renders the context into the question-judge prompt, caches the system prompt
        for the provider, and invokes the model with structured output, retrying on
        validation or parsing failures.

        Args:
            context: The question and its surrounding context to judge.

        Returns:
            The per-dimension and overall verdict on the question.

        Raises:
            JudgeError: If the model output fails validation or parsing, or is not a
                QuestionVerdict.
        """
        structured = self._model.with_structured_output(QuestionVerdict).with_retry(
            retry_if_exception_type=(ValidationError, OutputParserException),
            wait_exponential_jitter=False,
            stop_after_attempt=_JUDGE_RETRY_ATTEMPTS,
        )
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
        the model with structured output, retrying on validation or parsing failures.

        Args:
            target_level: The seniority level whose competencies are judged.
            transcript: The interview turns to assess coverage over.

        Returns:
            The per-competency and overall coverage verdict.

        Raises:
            JudgeError: If the model output fails validation or parsing, or is not a
                CoverageVerdict.
        """
        structured = self._model.with_structured_output(CoverageVerdict).with_retry(
            retry_if_exception_type=(ValidationError, OutputParserException),
            wait_exponential_jitter=False,
            stop_after_attempt=_JUDGE_RETRY_ATTEMPTS,
        )
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

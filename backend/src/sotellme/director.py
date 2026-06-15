from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

from sotellme.assessor import TopicAssessment
from sotellme.caching import cache_system_prompt
from sotellme.coverage import DEFAULT_FOLLOW_UP_CAP
from sotellme.interviewer import Turn, render_profile, render_transcript
from sotellme.profile import CandidateProfile
from sotellme.prompts import director_messages
from sotellme.role import RoleContext

DirectorAction = Literal["follow_up", "new_topic", "wrap_up", "terminate"]


class DirectorDecision(BaseModel):
    action: DirectorAction = Field(
        description=(
            "What happens next: follow_up digs into the last answer, new_topic opens "
            "a different stretch of the interview, wrap_up ends the session because "
            "there is enough signal, terminate ends it because the input has stopped "
            "being an interview."
        )
    )
    subject: str = Field(
        default="",
        description=(
            "For follow_up: the claim or aspect of the last answer to chase, "
            "near-verbatim. For new_topic: the topic to open, concrete enough that a "
            "colleague could ask about it. Empty for wrap_up and terminate."
        ),
    )
    reason: str = Field(
        default="",
        description=(
            "Why this is the right move now, in one sentence. Never shown to the candidate."
        ),
    )


@dataclass(frozen=True)
class DirectorSituation:
    profile: CandidateProfile
    context: RoleContext
    emphasis: tuple[str, ...]
    brief: str
    transcript: Sequence[Turn]
    assessments: Sequence[TopicAssessment]
    questions_asked: int
    question_cap: int
    consecutive_follow_ups: int = 0
    follow_up_cap: int = DEFAULT_FOLLOW_UP_CAP
    follow_ups_exhausted: bool = False


class DirectorError(Exception):
    pass


_DECIDE_FAILURE_MESSAGE = "Could not decide the next interview move. Try answering again."


def render_role_details(context: RoleContext) -> str:
    lines: list[str] = []
    if context.company:
        lines.append(f"Company: {context.company}")
    if context.role_title:
        lines.append(f"Role: {context.role_title}")
    if context.framework:
        lines.append(f"Values framework: {context.framework}")
    weighted = ", ".join(f"{c.name} ({c.weight})" for c in context.competencies)
    lines.append(f"Competencies the posting emphasizes: {weighted}")
    return "\n".join(lines)


def _star_gaps(entry: TopicAssessment) -> list[str]:
    star = entry.assessment.star
    missing = [
        name
        for name, present in (
            ("situation", star.situation),
            ("task", star.task),
            ("action", star.action),
            ("result", star.result),
        )
        if not present
    ]
    if star.result and not star.quantified_result:
        missing.append("a number on the result")
    return missing


def render_assessments(log: Sequence[TopicAssessment]) -> str:
    if not log:
        return "No answers assessed yet."
    lines: list[str] = []
    for index, entry in enumerate(log, start=1):
        signal = (
            "the topic holds enough signal"
            if entry.assessment.sufficient_signal
            else "the topic needs more signal"
        )
        line = f"After answer {index} (topic: {entry.topic}): {signal}"
        gaps = _star_gaps(entry)
        if gaps:
            line += f"; story elements not stated yet: {', '.join(gaps)}"
        if entry.assessment.claims_worth_chasing:
            line += f"; worth chasing: {', '.join(entry.assessment.claims_worth_chasing)}"
        lines.append(line)
    return "\n".join(lines)


class LLMDirector:
    def __init__(self, model: BaseChatModel, provider: str = "") -> None:
        self._model = model
        self._provider = provider

    def decide(self, situation: DirectorSituation) -> DirectorDecision:
        messages = cache_system_prompt(
            director_messages(
                role_details=render_role_details(situation.context),
                emphasis=situation.emphasis,
                brief=situation.brief,
                profile_text=render_profile(situation.profile),
                transcript_text=render_transcript(situation.transcript),
                assessment_notes=render_assessments(situation.assessments),
                questions_asked=situation.questions_asked,
                question_cap=situation.question_cap,
                consecutive_follow_ups=situation.consecutive_follow_ups,
                follow_up_cap=situation.follow_up_cap,
                follow_ups_exhausted=situation.follow_ups_exhausted,
            ),
            self._provider,
        )
        structured = self._model.with_structured_output(DirectorDecision)
        try:
            result = structured.invoke(messages)
        except (ValidationError, OutputParserException) as exc:
            raise DirectorError(_DECIDE_FAILURE_MESSAGE) from exc
        if not isinstance(result, DirectorDecision):
            raise DirectorError(_DECIDE_FAILURE_MESSAGE)
        return result

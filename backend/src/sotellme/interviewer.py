from collections.abc import Sequence
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from sotellme.profile import CandidateProfile
from sotellme.prompts import (
    FOLLOW_UP_DIRECTIVE_TEMPLATE,
    NEW_TOPIC_DIRECTIVE_TEMPLATE,
    closing_messages,
    question_messages,
    redirect_messages,
)
from sotellme.role import RoleContext
from sotellme.voice import sanitize

if TYPE_CHECKING:
    from sotellme.director import DirectorDecision


class Turn(BaseModel):
    question: str
    answer: str


def render_profile(profile: CandidateProfile) -> str:
    sections: list[str] = []
    if profile.roles:
        roles = [
            f"- {role.title}, {role.organization}" + (f" ({role.period})" if role.period else "")
            for role in profile.roles
        ]
        sections.append("Roles:\n" + "\n".join(roles))
    if profile.projects:
        projects = [f"- {project.name}: {project.description}" for project in profile.projects]
        sections.append("Projects:\n" + "\n".join(projects))
    if profile.quantified_claims:
        claims = [f"- {claim}" for claim in profile.quantified_claims]
        sections.append("Claims with numbers:\n" + "\n".join(claims))
    if profile.technologies:
        sections.append("Technologies: " + ", ".join(profile.technologies))
    return "\n".join(sections)


def render_transcript(transcript: Sequence[Turn]) -> str:
    return "\n".join(f"Q: {turn.question}\nA: {turn.answer}" for turn in transcript)


def render_role_context(context: RoleContext) -> str:
    lines: list[str] = []
    if context.company:
        lines.append(f"Company: {context.company}")
    if context.role_title:
        lines.append(f"Role: {context.role_title}")
    if context.framework:
        lines.append(f"Values framework: {context.framework}")
    return "\n".join(lines) or "The posting states no company or role name."


def render_directive(decision: "DirectorDecision") -> str:
    if decision.action == "follow_up":
        return FOLLOW_UP_DIRECTIVE_TEMPLATE.format(subject=decision.subject, reason=decision.reason)
    return NEW_TOPIC_DIRECTIVE_TEMPLATE.format(subject=decision.subject, reason=decision.reason)


class LLMInterviewer:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    def question_for(
        self,
        decision: "DirectorDecision",
        profile: CandidateProfile,
        context: RoleContext,
        brief: str,
        transcript: Sequence[Turn],
    ) -> str:
        messages = question_messages(
            role_details=render_role_context(context),
            brief=brief,
            profile_text=render_profile(profile),
            transcript_text=render_transcript(transcript),
            directive=render_directive(decision),
        )
        return sanitize(self._model.invoke(messages).text)

    def closing_turn(self, transcript: Sequence[Turn]) -> str:
        messages = closing_messages(render_transcript(transcript))
        return sanitize(self._model.invoke(messages).text)

    def redirect_turn(self, question: str) -> str:
        messages = redirect_messages(question)
        return sanitize(self._model.invoke(messages).text)

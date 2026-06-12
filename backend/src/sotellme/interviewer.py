from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from sotellme.coverage import Gap, MotivationTopic
from sotellme.profile import CandidateProfile
from sotellme.prompts import (
    closing_messages,
    competency_question_messages,
    motivation_question_messages,
    probe_question_messages,
)
from sotellme.role import RoleContext
from sotellme.voice import sanitize


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


class LLMInterviewer:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    def competency_question(
        self, profile: CandidateProfile, transcript: Sequence[Turn], competency: str
    ) -> str:
        messages = competency_question_messages(
            render_profile(profile), render_transcript(transcript), competency
        )
        return sanitize(self._model.invoke(messages).text)

    def motivation_question(
        self,
        context: RoleContext,
        posting_text: str,
        transcript: Sequence[Turn],
        topic: MotivationTopic,
    ) -> str:
        messages = motivation_question_messages(
            render_role_context(context), posting_text, render_transcript(transcript), topic
        )
        return sanitize(self._model.invoke(messages).text)

    def probe_question(
        self, profile: CandidateProfile, transcript: Sequence[Turn], gaps: tuple[Gap, ...]
    ) -> str:
        messages = probe_question_messages(
            render_profile(profile), render_transcript(transcript), gaps
        )
        return sanitize(self._model.invoke(messages).text)

    def closing_turn(self, transcript: Sequence[Turn]) -> str:
        messages = closing_messages(render_transcript(transcript))
        return sanitize(self._model.invoke(messages).text)

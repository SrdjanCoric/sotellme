from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from sotellme.coverage import Gap
from sotellme.profile import CandidateProfile
from sotellme.prompts import (
    closing_messages,
    opening_question_messages,
    probe_question_messages,
)


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


class LLMInterviewer:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    def opening_question(self, profile: CandidateProfile) -> str:
        messages = opening_question_messages(render_profile(profile))
        return self._model.invoke(messages).text.strip()

    def probe_question(
        self, profile: CandidateProfile, transcript: Sequence[Turn], gaps: tuple[Gap, ...]
    ) -> str:
        messages = probe_question_messages(
            render_profile(profile), render_transcript(transcript), gaps
        )
        return self._model.invoke(messages).text.strip()

    def closing_turn(self, transcript: Sequence[Turn]) -> str:
        messages = closing_messages(render_transcript(transcript))
        return self._model.invoke(messages).text.strip()

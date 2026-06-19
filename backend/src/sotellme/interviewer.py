"""Render interview context and generate interviewer turns from the model."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from sotellme.caching import cache_system_prompt
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
    """A single interview exchange."""

    question: str
    answer: str


def render_profile(profile: CandidateProfile) -> str:
    """Render a candidate profile into a plain-text block for the prompts."""
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
    """Render interview turns into a Q/A plain-text transcript."""
    return "\n".join(f"Q: {turn.question}\nA: {turn.answer}" for turn in transcript)


def render_role_context(context: RoleContext) -> str:
    """Render the role context into a plain-text block for the question prompt."""
    lines: list[str] = []
    if context.company:
        lines.append(f"Company: {context.company}")
    if context.role_title:
        lines.append(f"Role: {context.role_title}")
    if context.framework:
        lines.append(f"Values framework: {context.framework}")
    return "\n".join(lines) or "The posting states no company or role name."


def render_directive(decision: "DirectorDecision") -> str:
    """Render a director decision into a directive line for the question prompt."""
    if decision.action == "follow_up":
        return FOLLOW_UP_DIRECTIVE_TEMPLATE.format(subject=decision.subject, reason=decision.reason)
    if decision.action == "new_topic":
        return NEW_TOPIC_DIRECTIVE_TEMPLATE.format(subject=decision.subject, reason=decision.reason)
    raise ValueError(
        f"render_directive received the closing action {decision.action!r}; "
        "wrap_up and terminate route to pose_closing, not a directive."
    )


class LLMInterviewer:
    """Generate interviewer turns by prompting a chat model."""

    def __init__(self, model: BaseChatModel, provider: str = "") -> None:
        self._model = model
        self._provider = provider

    def question_for(
        self,
        decision: "DirectorDecision",
        profile: CandidateProfile,
        context: RoleContext,
        brief: str,
        transcript: Sequence[Turn],
    ) -> str:
        """Generate the next interviewer question for the given director decision.

        Renders the role context, brief, profile, transcript, and directive into the
        question prompt, caches the system prompt for the provider, invokes the model,
        and sanitizes the result.

        Args:
            decision: The director decision that drives the next question.
            profile: The candidate profile.
            context: The role context.
            brief: The interview brief text.
            transcript: The interview turns so far.

        Returns:
            The sanitized next question text.
        """
        messages = cache_system_prompt(
            question_messages(
                role_details=render_role_context(context),
                brief=brief,
                profile_text=render_profile(profile),
                transcript_text=render_transcript(transcript),
                directive=render_directive(decision),
            ),
            self._provider,
        )
        return sanitize(self._model.invoke(messages).text)

    def closing_turn(self, transcript: Sequence[Turn]) -> str:
        """Generate the interviewer's closing turn for the session.

        Args:
            transcript: The interview turns so far.

        Returns:
            The sanitized closing turn text.
        """
        messages = cache_system_prompt(
            closing_messages(render_transcript(transcript)), self._provider
        )
        return sanitize(self._model.invoke(messages).text)

    def redirect_turn(self, question: str) -> str:
        """Generate a redirect turn that steers the candidate back to the question.

        Args:
            question: The question the candidate should be redirected toward.

        Returns:
            The sanitized redirect turn text.
        """
        messages = cache_system_prompt(redirect_messages(question), self._provider)
        return sanitize(self._model.invoke(messages).text)

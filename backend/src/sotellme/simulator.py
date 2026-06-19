"""LLM-backed candidate simulator that role-plays personas during simulated interviews."""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel

from sotellme.caching import cache_system_prompt
from sotellme.interviewer import Turn, render_transcript
from sotellme.personas import Persona
from sotellme.prompts import BEHAVIOR_DIRECTIVES, candidate_simulator_messages


def render_persona(persona: Persona) -> str:
    """Render a persona into the text block the simulator prompt embeds."""
    return (
        f"Target level: {persona.target_level}\n"
        f"Answering profile: {persona.profile}\n\n"
        f"CV:\n{persona.cv}"
    )


class CandidateSimulator:
    """Role-play a candidate persona's answers using a chat model."""

    def __init__(self, model: BaseChatModel, provider: str = "") -> None:
        self._model = model
        self._provider = provider

    def answer(self, persona: Persona, question: str, transcript: Sequence[Turn]) -> str:
        """Generate the persona's spoken answer to a question for the current turn.

        Selects the answering behavior for this turn from the persona, builds the
        simulator messages with provider-specific system-prompt caching, and invokes
        the model.

        Args:
            persona: The persona being role-played.
            question: The interviewer's question to answer.
            transcript: The interview turns so far; its length determines the turn number.

        Returns:
            The text the candidate says out loud in reply.
        """
        turn = len(transcript) + 1
        behavior = persona.behavior_for(turn)
        messages = cache_system_prompt(
            candidate_simulator_messages(
                persona_text=render_persona(persona),
                behavior_directive=BEHAVIOR_DIRECTIVES[behavior],
                question=question,
                transcript_text=render_transcript(transcript),
            ),
            self._provider,
        )
        return self._model.invoke(messages).text

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel

from sotellme.caching import cache_system_prompt
from sotellme.interviewer import Turn, render_transcript
from sotellme.personas import Persona
from sotellme.prompts import BEHAVIOR_DIRECTIVES, candidate_simulator_messages


def render_persona(persona: Persona) -> str:
    return (
        f"Target level: {persona.target_level}\n"
        f"Answering profile: {persona.profile}\n\n"
        f"CV:\n{persona.cv}"
    )


class CandidateSimulator:
    def __init__(self, model: BaseChatModel, provider: str = "") -> None:
        self._model = model
        self._provider = provider

    def answer(self, persona: Persona, question: str, transcript: Sequence[Turn]) -> str:
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

"""Envelope checks that bound how many questions and follow-ups an interview may ask."""

from dataclasses import dataclass

DEFAULT_QUESTION_CAP = 20
DEFAULT_FOLLOW_UP_CAP = 6


@dataclass(frozen=True)
class EnvelopeState:
    """Snapshot of interview progress used to decide whether to keep asking."""

    questions_asked: int
    consecutive_follow_ups: int = 0
    budget_exhausted: bool = False
    vetoed: bool = False


def question_allowed(state: EnvelopeState, question_cap: int = DEFAULT_QUESTION_CAP) -> bool:
    """Decide whether another question may be asked."""
    if state.budget_exhausted or state.vetoed:
        return False
    return state.questions_asked < question_cap


def follow_up_allowed(state: EnvelopeState, follow_up_cap: int = DEFAULT_FOLLOW_UP_CAP) -> bool:
    """Decide whether another follow-up may be asked on the current thread."""
    return state.consecutive_follow_ups < follow_up_cap

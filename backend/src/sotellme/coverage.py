from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

Gap = Literal["situation", "task", "action", "result", "quantified_result"]


class StarFlags(BaseModel):
    situation: bool = Field(
        description="The answer sets the scene: the team, company, or context the story happens in."
    )
    task: bool = Field(
        description="The answer names a concrete problem or goal the candidate had to address."
    )
    action: bool = Field(description="The answer describes what the candidate actually did.")
    result: bool = Field(description="The answer states an outcome: how things ended up.")
    quantified_result: bool = Field(
        description="The stated outcome carries a number or other measurable change."
    )


@dataclass(frozen=True)
class Probe:
    gaps: tuple[Gap, ...]


@dataclass(frozen=True)
class NextCompetency:
    pass


@dataclass(frozen=True)
class Stop:
    pass


DEFAULT_FOLLOWUP_CAP = 3


@dataclass(frozen=True)
class CoverageState:
    flags: StarFlags
    followups_used: int = 0
    budget_exhausted: bool = False


def story_gaps(flags: StarFlags) -> tuple[Gap, ...]:
    gaps: list[Gap] = []
    if not flags.situation:
        gaps.append("situation")
    if not flags.task:
        gaps.append("task")
    if not flags.action:
        gaps.append("action")
    if not flags.result:
        gaps.append("result")
    elif not flags.quantified_result:
        gaps.append("quantified_result")
    return tuple(gaps)


def next_action(
    state: CoverageState, followup_cap: int = DEFAULT_FOLLOWUP_CAP
) -> Probe | NextCompetency | Stop:
    if state.budget_exhausted:
        return Stop()
    gaps = story_gaps(state.flags)
    if gaps and state.followups_used < followup_cap:
        return Probe(gaps=gaps)
    return NextCompetency()

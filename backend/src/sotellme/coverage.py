from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from sotellme.role import RoleContext

Gap = Literal["situation", "task", "action", "result", "quantified_result"]

DEFAULT_MAX_COMPETENCIES = 5


def plan_competencies(
    context: RoleContext, limit: int = DEFAULT_MAX_COMPETENCIES
) -> tuple[str, ...]:
    ranked = sorted(context.competencies, key=lambda c: -c.weight)
    return tuple(c.name for c in ranked[:limit])


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


MotivationTopic = Literal["company", "role"]

MOTIVATION_TOPICS: tuple[MotivationTopic, ...] = ("company", "role")


@dataclass(frozen=True)
class Probe:
    gaps: tuple[Gap, ...]


@dataclass(frozen=True)
class NextCompetency:
    competency: str


@dataclass(frozen=True)
class Motivation:
    topic: MotivationTopic


@dataclass(frozen=True)
class Stop:
    pass


DEFAULT_FOLLOWUP_CAP = 3


@dataclass(frozen=True)
class CoverageState:
    flags: StarFlags | None = None
    followups_used: int = 0
    competencies_remaining: tuple[str, ...] = ()
    motivation_remaining: tuple[MotivationTopic, ...] = ()
    in_motivation: bool = False
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
) -> Probe | NextCompetency | Motivation | Stop:
    if state.budget_exhausted:
        return Stop()
    if not state.in_motivation:
        gaps = story_gaps(state.flags) if state.flags is not None else ()
        if gaps and state.followups_used < followup_cap:
            return Probe(gaps=gaps)
        if state.competencies_remaining:
            return NextCompetency(competency=state.competencies_remaining[0])
    if state.motivation_remaining:
        return Motivation(topic=state.motivation_remaining[0])
    return Stop()

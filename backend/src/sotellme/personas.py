"""Define evaluation personas and their per-turn answer behaviors loaded from JSON."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from sotellme.role import TargetLevel

AnswerBehavior = Literal[
    "complete_star",
    "thin",
    "blurred_ownership",
    "off_topic",
    "rambling",
    "confident_bluffer",
    "inappropriate",
    "strong_terse",
]


class PlantedTurn(BaseModel):
    """An answer behavior override for a specific question turn."""

    turn: int = Field(ge=1, description="The 1-based question turn this override applies to.")
    behavior: AnswerBehavior


class Persona(BaseModel):
    """A scripted candidate persona for driving an evaluation run."""

    name: str
    target_level: TargetLevel
    cv: str
    posting: str
    profile: str
    base_behavior: AnswerBehavior
    planted_turns: list[PlantedTurn] = Field(default_factory=list)

    def behavior_for(self, turn: int) -> AnswerBehavior:
        """Return the answer behavior for a question turn."""
        for planted in self.planted_turns:
            if planted.turn == turn:
                return planted.behavior
        return self.base_behavior


def load_personas(personas_dir: Path) -> list[Persona]:
    """Load and validate all persona JSON files from a directory."""
    return [
        Persona.model_validate(json.loads(path.read_text()))
        for path in sorted(personas_dir.glob("*.json"))
    ]

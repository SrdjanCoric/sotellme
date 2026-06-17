from dataclasses import dataclass
from typing import Any

from sotellme.eval_datasets import apply_limit
from sotellme.personas import Persona
from sotellme.sim_datasets import select_persona_items


@dataclass
class _Item:
    input: dict[str, Any]


def _persona(name: str) -> dict[str, Any]:
    return Persona.model_validate(
        {
            "name": name,
            "target_level": "senior",
            "cv": "cv",
            "posting": "posting",
            "profile": "profile",
            "base_behavior": "complete_star",
        }
    ).model_dump()


_ITEMS = [_Item(input=_persona("senior-strong")), _Item(input=_persona("junior-thin"))]


def test_select_persona_items_returns_everything_when_no_names_given() -> None:
    assert select_persona_items(_ITEMS, None) == _ITEMS
    assert select_persona_items(_ITEMS, set()) == _ITEMS


def test_select_persona_items_keeps_only_the_named_personas() -> None:
    selected = select_persona_items(_ITEMS, {"senior-strong"})

    assert [item.input["name"] for item in selected] == ["senior-strong"]


def test_apply_limit_of_zero_selects_no_personas() -> None:
    assert apply_limit(_ITEMS, 0) == []


def test_apply_limit_of_none_runs_every_persona() -> None:
    assert apply_limit(_ITEMS, None) == _ITEMS

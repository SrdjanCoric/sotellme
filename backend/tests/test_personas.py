import json
from pathlib import Path
from typing import get_args

from sotellme.personas import AnswerBehavior, Persona, load_personas
from sotellme.role import TargetLevel

PERSONAS_DIR = Path(__file__).parent.parent / "evals" / "personas"


def _persona(**overrides: object) -> Persona:
    base: dict[str, object] = {
        "name": "senior-strong",
        "target_level": "senior",
        "cv": "Mira Stanic, senior engineer.",
        "posting": "Senior backend engineer.",
        "profile": "Speaks in concrete metrics, credits the team honestly.",
        "base_behavior": "complete_star",
        "planted_turns": [],
    }
    base.update(overrides)
    return Persona.model_validate(base)


def test_a_persona_is_not_expected_to_terminate_by_default() -> None:
    assert _persona().expected_to_terminate is False


def test_a_persona_can_be_marked_expected_to_terminate() -> None:
    assert _persona(expected_to_terminate=True).expected_to_terminate is True


def test_behavior_for_falls_back_to_the_base_behavior() -> None:
    persona = _persona(base_behavior="thin")

    assert persona.behavior_for(1) == "thin"
    assert persona.behavior_for(7) == "thin"


def test_a_planted_turn_overrides_the_base_behavior_on_that_turn() -> None:
    persona = _persona(
        base_behavior="complete_star",
        planted_turns=[
            {"turn": 3, "behavior": "off_topic"},
            {"turn": 5, "behavior": "inappropriate"},
        ],
    )

    assert persona.behavior_for(2) == "complete_star"
    assert persona.behavior_for(3) == "off_topic"
    assert persona.behavior_for(5) == "inappropriate"
    assert persona.behavior_for(6) == "complete_star"


def test_load_personas_reads_committed_json_files_sorted_by_name(tmp_path: Path) -> None:
    (tmp_path / "senior-strong.json").write_text(
        json.dumps(
            {
                "name": "senior-strong",
                "target_level": "senior",
                "cv": "Mira Stanic, senior engineer.",
                "posting": "Senior backend engineer.",
                "profile": "Concrete, owns outcomes.",
                "base_behavior": "complete_star",
            }
        )
    )
    (tmp_path / "junior-thin.json").write_text(
        json.dumps(
            {
                "name": "junior-thin",
                "target_level": "junior",
                "cv": "Junior dev.",
                "posting": "Junior backend engineer.",
                "profile": "Vague, light on detail.",
                "base_behavior": "thin",
            }
        )
    )

    personas = load_personas(tmp_path)

    assert [p.name for p in personas] == ["junior-thin", "senior-strong"]
    assert personas[0].target_level == "junior"
    assert isinstance(personas[1], Persona)


def test_committed_personas_cover_every_level_and_the_full_answer_mixture() -> None:
    personas = load_personas(PERSONAS_DIR)

    levels = {p.target_level for p in personas}
    assert levels == set(get_args(TargetLevel))

    seen: set[str] = set()
    for persona in personas:
        seen.add(persona.base_behavior)
        seen.update(planted.behavior for planted in persona.planted_turns)
    assert seen == set(get_args(AnswerBehavior))


def test_committed_personas_have_off_topic_and_inappropriate_guardrail_fixtures() -> None:
    personas = load_personas(PERSONAS_DIR)

    base_behaviors = {p.base_behavior for p in personas}
    assert "off_topic" in base_behaviors
    assert "inappropriate" in base_behaviors


def test_a_termination_persona_plants_no_recovery_turn() -> None:
    personas = load_personas(PERSONAS_DIR)

    for persona in personas:
        if not persona.expected_to_terminate:
            continue
        # Under the terminate policy the interview never survives to a planted recovery turn,
        # so a termination persona must plant none.
        assert persona.planted_turns == [], (
            f"termination persona {persona.name!r} plants a turn it never reaches"
        )


def test_the_committed_termination_personas_are_the_expected_pair() -> None:
    personas = load_personas(PERSONAS_DIR)

    marked = {p.name for p in personas if p.expected_to_terminate}
    # staff-injection exercises manipulation; mid-offtopic exercises persistent off-topic.
    assert marked == {"staff-injection", "mid-offtopic"}


def test_no_coverage_persona_carries_a_turn_the_guardrail_terminates_on() -> None:
    # A coverage persona must run full-length to be judged on competency coverage. Both an
    # `inappropriate` turn and an `off_topic` turn trip the Task 0031 policy: inappropriate
    # terminates on the first screen, and an off-topic reply is re-posed on the redirect and
    # drifts again into a terminating second consecutive off-topic. Either truncates the
    # transcript, so only a persona built to terminate (expected_to_terminate) may carry one.
    personas = load_personas(PERSONAS_DIR)

    for persona in personas:
        if persona.expected_to_terminate:
            continue
        behaviors = {
            persona.base_behavior,
            *(planted.behavior for planted in persona.planted_turns),
        }
        terminating = behaviors & {"inappropriate", "off_topic"}
        assert not terminating, (
            f"coverage persona {persona.name!r} carries {terminating} the guardrail terminates on"
        )

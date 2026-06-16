from typing import get_args

from stubs import StubChatModel

from sotellme.interviewer import Turn
from sotellme.personas import AnswerBehavior, Persona
from sotellme.prompts import BEHAVIOR_DIRECTIVES
from sotellme.simulator import CandidateSimulator


def _persona(**overrides: object) -> Persona:
    base: dict[str, object] = {
        "name": "senior-strong",
        "target_level": "senior",
        "cv": "Naoki Brennan, senior engineer. Led the route-optimization rewrite.",
        "posting": "Senior backend engineer.",
        "profile": "Tells complete, quantified STAR stories and owns his decisions.",
        "base_behavior": "complete_star",
        "planted_turns": [],
    }
    base.update(overrides)
    return Persona.model_validate(base)


def _prompt_text(model: StubChatModel) -> str:
    return " ".join(text for _, text in model.seen_inputs[0])


def test_the_simulator_returns_the_models_answer() -> None:
    model = StubChatModel(text_response="At Arcwell I led the rewrite and cut planning to 25s.")

    answer = CandidateSimulator(model).answer(
        _persona(), question="Tell me about a project you're proud of.", transcript=[]
    )

    assert answer == "At Arcwell I led the rewrite and cut planning to 25s."


def test_the_simulator_performs_the_base_behavior_when_no_turn_is_planted() -> None:
    model = StubChatModel(text_response="...")

    CandidateSimulator(model).answer(
        _persona(base_behavior="thin"), question="Walk me through it.", transcript=[]
    )

    assert BEHAVIOR_DIRECTIVES["thin"] in _prompt_text(model)
    assert BEHAVIOR_DIRECTIVES["complete_star"] not in _prompt_text(model)


def test_a_planted_turn_switches_the_behavior_for_that_turn() -> None:
    model = StubChatModel(text_response="...")
    persona = _persona(
        base_behavior="complete_star",
        planted_turns=[{"turn": 3, "behavior": "inappropriate"}],
    )
    transcript = [
        Turn(question="Q1", answer="A1"),
        Turn(question="Q2", answer="A2"),
    ]

    CandidateSimulator(model).answer(persona, question="Q3?", transcript=transcript)

    assert BEHAVIOR_DIRECTIVES["inappropriate"] in _prompt_text(model)
    assert BEHAVIOR_DIRECTIVES["complete_star"] not in _prompt_text(model)


def test_the_simulator_sees_the_persona_question_and_transcript() -> None:
    model = StubChatModel(text_response="...")
    transcript = [Turn(question="Earlier question", answer="Earlier answer")]

    CandidateSimulator(model).answer(
        _persona(), question="What was the hardest trade-off?", transcript=transcript
    )

    prompt = _prompt_text(model)
    assert "Naoki Brennan" in prompt
    assert "route-optimization rewrite" in prompt
    assert "What was the hardest trade-off?" in prompt
    assert "Earlier answer" in prompt


def test_every_answer_behavior_has_a_directive() -> None:
    assert set(BEHAVIOR_DIRECTIVES) == set(get_args(AnswerBehavior))

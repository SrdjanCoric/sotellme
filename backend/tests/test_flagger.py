import pytest
from stubs import StubChatModel

from sotellme.coverage import StarFlags
from sotellme.flagger import StarFlaggerError, flag_star_elements

COMPLETE_FLAGS = StarFlags(
    situation=True, task=True, action=True, result=True, quantified_result=True
)


def test_flagger_returns_the_structured_flags() -> None:
    model = StubChatModel(structured_response=COMPLETE_FLAGS)

    flags = flag_star_elements("We migrated the pipeline and cut latency 38%.", model)

    assert flags == COMPLETE_FLAGS


def test_flagger_sends_the_answer_to_the_model_as_delimited_data() -> None:
    model = StubChatModel(structured_response=COMPLETE_FLAGS)

    flag_star_elements("We migrated the pipeline.", model)

    human_texts = [text for role, text in model.seen_inputs[0] if role == "human"]
    assert "<answer>\nWe migrated the pipeline.\n</answer>" in human_texts[0]


def test_flagging_that_returns_nothing_is_a_clear_error() -> None:
    model = StubChatModel(structured_response=None)

    with pytest.raises(StarFlaggerError, match="Could not read the answer"):
        flag_star_elements("An answer.", model)

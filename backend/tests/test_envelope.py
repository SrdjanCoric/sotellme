from sotellme.coverage import (
    DEFAULT_FOLLOW_UP_CAP,
    DEFAULT_QUESTION_CAP,
    EnvelopeState,
    follow_up_allowed,
    question_allowed,
)


def test_under_the_cap_another_question_is_allowed() -> None:
    assert question_allowed(EnvelopeState(questions_asked=3))


def test_the_hard_question_cap_forces_the_close() -> None:
    state = EnvelopeState(questions_asked=DEFAULT_QUESTION_CAP)

    assert not question_allowed(state)


def test_the_cap_is_configurable() -> None:
    assert question_allowed(EnvelopeState(questions_asked=7), question_cap=8)
    assert not question_allowed(EnvelopeState(questions_asked=8), question_cap=8)


def test_under_the_follow_up_cap_another_follow_up_is_allowed() -> None:
    state = EnvelopeState(questions_asked=6, consecutive_follow_ups=DEFAULT_FOLLOW_UP_CAP - 1)

    assert follow_up_allowed(state)


def test_the_follow_up_cap_forces_a_topic_change() -> None:
    state = EnvelopeState(questions_asked=6, consecutive_follow_ups=DEFAULT_FOLLOW_UP_CAP)

    assert not follow_up_allowed(state)


def test_the_follow_up_cap_is_configurable() -> None:
    state = EnvelopeState(questions_asked=6, consecutive_follow_ups=2)

    assert follow_up_allowed(state, follow_up_cap=3)
    assert not follow_up_allowed(state, follow_up_cap=2)


def test_an_exhausted_budget_forces_the_close() -> None:
    state = EnvelopeState(questions_asked=2, budget_exhausted=True)

    assert not question_allowed(state)


def test_a_guardrail_veto_forces_the_close() -> None:
    state = EnvelopeState(questions_asked=2, vetoed=True)

    assert not question_allowed(state)

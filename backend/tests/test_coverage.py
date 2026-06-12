from sotellme.coverage import CoverageState, NextCompetency, Probe, StarFlags, Stop, next_action


def flags(
    situation: bool = True,
    task: bool = True,
    action: bool = True,
    result: bool = True,
    quantified_result: bool = True,
) -> StarFlags:
    return StarFlags(
        situation=situation,
        task=task,
        action=action,
        result=result,
        quantified_result=quantified_result,
    )


def test_a_complete_quantified_story_advances_to_the_next_competency() -> None:
    action = next_action(CoverageState(flags=flags()))

    assert action == NextCompetency()


def test_missing_elements_are_probed_in_narrative_order() -> None:
    action = next_action(CoverageState(flags=flags(task=False, result=False)))

    assert action == Probe(gaps=("task", "result"))


def test_an_unquantified_result_is_probed_for_the_number() -> None:
    action = next_action(CoverageState(flags=flags(quantified_result=False)))

    assert action == Probe(gaps=("quantified_result",))


def test_a_missing_result_is_probed_as_result_not_quantification() -> None:
    action = next_action(CoverageState(flags=flags(result=False, quantified_result=False)))

    assert action == Probe(gaps=("result",))


def test_an_incomplete_story_still_probes_below_the_followup_cap() -> None:
    state = CoverageState(flags=flags(result=False), followups_used=2)

    assert next_action(state, followup_cap=3) == Probe(gaps=("result",))


def test_the_followup_cap_advances_an_incomplete_story_without_belaboring() -> None:
    state = CoverageState(flags=flags(result=False), followups_used=3)

    assert next_action(state, followup_cap=3) == NextCompetency()


def test_an_exhausted_budget_stops_even_with_an_incomplete_story() -> None:
    state = CoverageState(flags=flags(result=False), budget_exhausted=True)

    assert next_action(state) == Stop()


def test_an_exhausted_budget_stops_even_with_a_complete_story() -> None:
    state = CoverageState(flags=flags(), budget_exhausted=True)

    assert next_action(state) == Stop()

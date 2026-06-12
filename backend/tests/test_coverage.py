from sotellme.coverage import (
    MOTIVATION_TOPICS,
    CoverageState,
    Motivation,
    NextCompetency,
    Probe,
    StarFlags,
    Stop,
    next_action,
    plan_competencies,
)
from sotellme.role import CompetencyWeight, RoleContext


def context_with(weights: dict[str, int]) -> RoleContext:
    return RoleContext(
        competencies=[CompetencyWeight(name=name, weight=w) for name, w in weights.items()],
    )


def test_the_plan_walks_competencies_by_weight_breadth_first() -> None:
    context = context_with({"conflict": 2, "ownership": 5, "impact": 4})

    assert plan_competencies(context) == ("ownership", "impact", "conflict")


def test_the_plan_keeps_posting_order_on_equal_weights() -> None:
    context = context_with({"conflict": 3, "ownership": 3, "impact": 3})

    assert plan_competencies(context) == ("conflict", "ownership", "impact")


def test_the_plan_is_capped_at_the_session_envelope() -> None:
    weights = {f"principle {i}": 5 - i % 3 for i in range(14)}
    context = context_with(weights)

    plan = plan_competencies(context, limit=5)

    assert len(plan) == 5
    assert plan == ("principle 0", "principle 3", "principle 6", "principle 9", "principle 12")


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
    state = CoverageState(flags=flags(), competencies_remaining=("conflict", "failure"))

    assert next_action(state) == NextCompetency(competency="conflict")


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
    state = CoverageState(
        flags=flags(result=False), followups_used=3, competencies_remaining=("failure",)
    )

    assert next_action(state, followup_cap=3) == NextCompetency(competency="failure")


def test_the_last_story_hands_over_to_the_motivation_segment() -> None:
    state = CoverageState(flags=flags(), motivation_remaining=MOTIVATION_TOPICS)

    assert next_action(state) == Motivation(topic="company")


def test_a_motivation_answer_advances_to_the_remaining_topic() -> None:
    state = CoverageState(in_motivation=True, motivation_remaining=("role",))

    assert next_action(state) == Motivation(topic="role")


def test_the_session_stops_when_stories_and_motivation_are_done() -> None:
    state = CoverageState(flags=flags(), competencies_remaining=(), motivation_remaining=())

    assert next_action(state) == Stop()


def test_without_a_posting_the_session_stops_after_the_stories() -> None:
    state = CoverageState(flags=flags(result=False), followups_used=3)

    assert next_action(state, followup_cap=3) == Stop()


def test_motivation_answers_are_never_probed() -> None:
    state = CoverageState(
        flags=flags(result=False), in_motivation=True, motivation_remaining=("role",)
    )

    assert next_action(state) == Motivation(topic="role")


def test_an_exhausted_budget_stops_even_with_an_incomplete_story() -> None:
    state = CoverageState(flags=flags(result=False), budget_exhausted=True)

    assert next_action(state) == Stop()


def test_an_exhausted_budget_stops_even_with_a_complete_story() -> None:
    state = CoverageState(flags=flags(), budget_exhausted=True)

    assert next_action(state) == Stop()

from sotellme.catalog import ModelPrice, default_catalog
from sotellme.pricing import (
    ModelUsage,
    cost_usd,
    estimate_session_cost,
    merge_usage,
    summarize_actual_cost,
)

PRICES = {"cheap": ModelPrice(input=1.0, output=2.0)}


def test_merge_usage_sums_per_model_token_counts_across_snapshots() -> None:
    one = {"opus": ModelUsage(input_tokens=100, output_tokens=20, cached_input_tokens=10)}
    two = {"opus": ModelUsage(input_tokens=80, output_tokens=30, cached_input_tokens=5)}

    assert merge_usage([one, two]) == {
        "opus": ModelUsage(input_tokens=180, output_tokens=50, cached_input_tokens=15)
    }


def test_merge_usage_keeps_models_that_appear_in_only_one_snapshot() -> None:
    one = {"opus": ModelUsage(input_tokens=100, output_tokens=20)}
    two = {"sonnet": ModelUsage(input_tokens=40, output_tokens=15)}

    assert merge_usage([one, two]) == {
        "opus": ModelUsage(input_tokens=100, output_tokens=20),
        "sonnet": ModelUsage(input_tokens=40, output_tokens=15),
    }


def test_merge_usage_of_nothing_is_empty() -> None:
    assert merge_usage([]) == {}


def default_prices() -> dict[str, ModelPrice]:
    return default_catalog().prices


def test_cached_input_defaults_to_the_full_input_rate() -> None:
    assert ModelPrice(input=3.0, output=15.0).cached_input_rate == 3.0


def test_cached_input_can_be_set_below_the_input_rate() -> None:
    assert ModelPrice(input=3.0, output=15.0, cached_input=0.3).cached_input_rate == 0.3


def test_cost_combines_input_and_output_rates_per_million_tokens() -> None:
    price = ModelPrice(input=3.0, output=15.0)

    assert cost_usd(price, input_tokens=1_000_000, output_tokens=1_000_000) == 18.0


def test_cost_scales_with_partial_millions() -> None:
    price = ModelPrice(input=2.0, output=12.0)

    assert cost_usd(price, input_tokens=500_000, output_tokens=100_000) == 1.0 + 1.2


def test_cached_input_tokens_are_billed_at_the_reduced_rate() -> None:
    price = ModelPrice(input=10.0, output=20.0, cached_input=1.0)

    full = cost_usd(price, input_tokens=1_000_000, output_tokens=0)
    discounted = cost_usd(
        price, input_tokens=1_000_000, output_tokens=0, cached_input_tokens=1_000_000
    )

    assert full == 10.0
    assert discounted == 1.0


def test_only_the_cached_portion_of_input_is_discounted() -> None:
    price = ModelPrice(input=10.0, output=20.0, cached_input=1.0)

    cost = cost_usd(price, input_tokens=1_000_000, output_tokens=0, cached_input_tokens=400_000)

    assert cost == (600_000 * 10.0 + 400_000 * 1.0) / 1_000_000


def test_cost_never_goes_negative_when_cached_exceeds_input() -> None:
    price = ModelPrice(input=10.0, output=20.0, cached_input=1.0)

    cost = cost_usd(price, input_tokens=100, output_tokens=0, cached_input_tokens=1_000)

    assert cost == 1_000 * 1.0 / 1_000_000
    assert cost >= 0


def test_a_longer_interview_is_estimated_to_cost_more() -> None:
    short = estimate_session_cost("cheap", expected_turns=5, prices=PRICES)
    longer = estimate_session_cost("cheap", expected_turns=15, prices=PRICES)

    assert longer.input_tokens > short.input_tokens
    assert longer.output_tokens > short.output_tokens
    assert longer.usd is not None and short.usd is not None
    assert longer.usd > short.usd


def test_the_estimate_prices_its_token_counts_with_the_named_models_rate() -> None:
    estimate = estimate_session_cost("cheap", expected_turns=10, prices=PRICES)

    assert estimate.usd == cost_usd(PRICES["cheap"], estimate.input_tokens, estimate.output_tokens)


def test_an_unpriced_model_estimates_tokens_but_not_dollars() -> None:
    estimate = estimate_session_cost("mystery", expected_turns=10, prices=PRICES)

    assert estimate.input_tokens > 0
    assert estimate.usd is None


def test_the_estimate_defaults_to_the_packaged_price_map() -> None:
    estimate = estimate_session_cost("claude-opus-4-8", expected_turns=10)

    assert estimate.usd is not None and estimate.usd > 0


def test_the_summary_totals_tokens_and_dollars_across_models() -> None:
    usage = {
        "claude-sonnet-4-6": ModelUsage(input_tokens=1_000_000, output_tokens=0),
        "claude-opus-4-8": ModelUsage(input_tokens=0, output_tokens=1_000_000),
    }

    summary = summarize_actual_cost(usage, default_prices())

    assert summary.total_tokens == 2_000_000
    assert summary.usd == 3.0 + 25.0


def test_the_summary_breaks_tokens_and_cost_down_per_model_and_direction() -> None:
    prices = default_prices()
    usage = {
        "claude-sonnet-4-6": ModelUsage(input_tokens=120_000, output_tokens=9_000),
        "claude-opus-4-8": ModelUsage(input_tokens=8_000, output_tokens=4_000),
    }

    summary = summarize_actual_cost(usage, prices)

    by_model = {entry.model: entry for entry in summary.per_model}
    assert by_model["claude-sonnet-4-6"].input_tokens == 120_000
    assert by_model["claude-sonnet-4-6"].output_tokens == 9_000
    assert by_model["claude-sonnet-4-6"].usd == cost_usd(
        prices["claude-sonnet-4-6"], 120_000, 9_000
    )
    assert by_model["claude-opus-4-8"].usd == cost_usd(prices["claude-opus-4-8"], 8_000, 4_000)


def test_per_model_lines_are_ordered_by_model_name() -> None:
    usage = {
        "claude-opus-4-8": ModelUsage(input_tokens=1, output_tokens=1),
        "claude-sonnet-4-6": ModelUsage(input_tokens=1, output_tokens=1),
    }

    summary = summarize_actual_cost(usage, default_prices())

    assert [entry.model for entry in summary.per_model] == [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
    ]


def test_the_summary_counts_unpriced_tokens_but_leaves_their_cost_unknown() -> None:
    usage = {"mystery": ModelUsage(input_tokens=500, output_tokens=500)}

    summary = summarize_actual_cost(usage, default_prices())

    assert summary.total_tokens == 1_000
    assert summary.usd == 0.0
    assert summary.per_model[0].model == "mystery"
    assert summary.per_model[0].usd is None


def test_an_empty_session_summarizes_to_zero() -> None:
    summary = summarize_actual_cost({}, default_prices())

    assert summary.total_tokens == 0
    assert summary.usd == 0.0
    assert summary.saved_usd == 0.0
    assert summary.per_model == ()


def test_the_summary_prices_cached_input_at_the_reduced_rate() -> None:
    prices = {"m": ModelPrice(input=10.0, output=20.0, cached_input=1.0)}
    usage = {
        "m": ModelUsage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=1_000_000)
    }

    summary = summarize_actual_cost(usage, prices)

    assert summary.usd == 1.0


def test_the_summary_reports_cached_tokens_and_estimated_savings() -> None:
    prices = {"m": ModelPrice(input=10.0, output=20.0, cached_input=1.0)}
    usage = {"m": ModelUsage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=400_000)}

    summary = summarize_actual_cost(usage, prices)

    assert summary.per_model[0].cached_input_tokens == 400_000
    assert summary.saved_usd == 400_000 * (10.0 - 1.0) / 1_000_000


def test_an_unpriced_model_contributes_no_savings() -> None:
    usage = {"mystery": ModelUsage(input_tokens=1_000, output_tokens=0, cached_input_tokens=500)}

    summary = summarize_actual_cost(usage, default_prices())

    assert summary.saved_usd == 0.0
    assert summary.per_model[0].cached_input_tokens == 500

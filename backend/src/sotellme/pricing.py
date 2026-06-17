from collections.abc import Mapping
from dataclasses import dataclass

from sotellme.catalog import ModelPrice, default_catalog

# A behavioral interview is roughly a fixed setup pass (CV parse, role context, research),
# a handful of agent calls per question turn (director, interviewer, assessor, guardrail,
# each seeing the growing transcript), and a fixed feedback pass at the end (grader, coach).
# These per-stage token sizes are deliberate estimates calibrated 2026-06-15 against a real
# ~212k-token Gemini run (~2:1 input:output): a typical interview lands around 270k and a
# full-length one (the question cap) around 400k, matching the budget cap so the estimate is
# conservative rather than rosy. The end-of-session summary reports the real counts.
SETUP_INPUT_TOKENS = 18_000
SETUP_OUTPUT_TOKENS = 2_500
PER_TURN_INPUT_TOKENS = 11_000
PER_TURN_OUTPUT_TOKENS = 5_500
FEEDBACK_INPUT_TOKENS = 30_000
FEEDBACK_OUTPUT_TOKENS = 22_000

# A representative interview length for the up-front cost estimate; the director shapes
# the real count, and the end-of-session summary reports what was actually spent.
TYPICAL_TURNS = 12


@dataclass(frozen=True)
class CostEstimate:
    model: str
    expected_turns: int
    input_tokens: int
    output_tokens: int
    usd: float | None


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0


@dataclass(frozen=True)
class ModelCost:
    model: str
    input_tokens: int
    output_tokens: int
    usd: float | None
    cached_input_tokens: int = 0


@dataclass(frozen=True)
class CostSummary:
    per_model: tuple[ModelCost, ...]
    total_tokens: int
    usd: float
    saved_usd: float = 0.0


def cost_usd(
    price: ModelPrice,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    uncached_input = max(0, input_tokens - cached_input_tokens)
    return (
        uncached_input * price.input
        + cached_input_tokens * price.cached_input_rate
        + output_tokens * price.output
    ) / 1_000_000


def expected_session_tokens(expected_turns: int) -> tuple[int, int]:
    input_tokens = (
        SETUP_INPUT_TOKENS + expected_turns * PER_TURN_INPUT_TOKENS + FEEDBACK_INPUT_TOKENS
    )
    output_tokens = (
        SETUP_OUTPUT_TOKENS + expected_turns * PER_TURN_OUTPUT_TOKENS + FEEDBACK_OUTPUT_TOKENS
    )
    return input_tokens, output_tokens


def estimate_session_cost(
    model: str,
    expected_turns: int,
    prices: Mapping[str, ModelPrice] | None = None,
) -> CostEstimate:
    if prices is None:
        prices = default_catalog().prices
    input_tokens, output_tokens = expected_session_tokens(expected_turns)
    usd = cost_usd(prices[model], input_tokens, output_tokens) if model in prices else None
    return CostEstimate(
        model=model,
        expected_turns=expected_turns,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=usd,
    )


def summarize_actual_cost(
    usage: Mapping[str, ModelUsage], prices: Mapping[str, ModelPrice]
) -> CostSummary:
    per_model: list[ModelCost] = []
    total_tokens = 0
    usd = 0.0
    saved_usd = 0.0
    for model in sorted(usage):
        used = usage[model]
        total_tokens += used.input_tokens + used.output_tokens
        price = prices.get(model)
        if price is not None:
            model_usd: float | None = cost_usd(
                price, used.input_tokens, used.output_tokens, used.cached_input_tokens
            )
            saved_usd += used.cached_input_tokens * (price.input - price.cached_input_rate) / 1e6
        else:
            model_usd = None
        if model_usd is not None:
            usd += model_usd
        per_model.append(
            ModelCost(
                model=model,
                input_tokens=used.input_tokens,
                output_tokens=used.output_tokens,
                usd=model_usd,
                cached_input_tokens=used.cached_input_tokens,
            )
        )
    return CostSummary(
        per_model=tuple(per_model), total_tokens=total_tokens, usd=usd, saved_usd=saved_usd
    )


def format_cost_summary(summary: CostSummary) -> str:
    lines = [
        f"Tokens used: {summary.total_tokens:,} · estimated cost: ${summary.usd:.2f} (estimate)."
    ]
    if summary.saved_usd > 0:
        lines.append(f"Prompt caching saved about ${summary.saved_usd:.2f} (estimate).")
    for entry in summary.per_model:
        cost = f"${entry.usd:.2f}" if entry.usd is not None else "price not configured"
        cached = f" ({entry.cached_input_tokens:,} cached)" if entry.cached_input_tokens else ""
        lines.append(
            f"  {entry.model}: {entry.input_tokens:,} in{cached} / "
            f"{entry.output_tokens:,} out · {cost}"
        )
    return "\n".join(lines)

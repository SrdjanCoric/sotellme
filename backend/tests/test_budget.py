from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from sotellme.budget import (
    DEFAULT_TOKEN_BUDGET,
    RESERVE_FRACTION,
    BudgetCallback,
    SessionBudget,
    default_session_budget,
)
from sotellme.pricing import (
    FEEDBACK_INPUT_TOKENS,
    FEEDBACK_OUTPUT_TOKENS,
    TYPICAL_TURNS,
    expected_session_tokens,
)


def llm_result(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0) -> LLMResult:
    usage_metadata: dict[str, object] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cache_read:
        usage_metadata["input_token_details"] = {"cache_read": cache_read}
    message = AIMessage(
        content="",
        usage_metadata=usage_metadata,
        response_metadata={"model_name": model},
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_recording_tokens_accumulates_usage() -> None:
    budget = SessionBudget(total_budget=100_000, reserve=10_000)

    budget = budget.record(3_000).record(2_000)

    assert budget.tokens_used == 5_000


def test_a_fresh_budget_is_not_exhausted() -> None:
    assert not SessionBudget(total_budget=100_000, reserve=10_000).exhausted


def test_the_wrap_fires_at_85_percent_of_the_interview_allowance() -> None:
    # 100k total minus a 20k reserve leaves 80k for the interview; the wrap is at 85% of that.
    budget = SessionBudget(total_budget=100_000, reserve=20_000)

    assert not budget.record(67_999).exhausted
    assert budget.record(68_000).exhausted


def test_the_reserve_is_held_back_from_the_interview_allowance() -> None:
    budget = SessionBudget(total_budget=100_000, reserve=20_000)

    assert budget.interview_allowance == 80_000


def test_the_wrap_leaves_room_before_the_reserve_is_touched() -> None:
    # Once exhausted, what is still unspent must cover at least the reserved feedback budget.
    budget = SessionBudget(total_budget=100_000, reserve=20_000).record(68_000)

    assert budget.exhausted
    assert budget.total_budget - budget.tokens_used >= budget.reserve


def test_a_budget_with_no_reserve_still_wraps_under_the_total() -> None:
    budget = SessionBudget(total_budget=10_000, reserve=0)

    assert not budget.record(8_499).exhausted
    assert budget.record(8_500).exhausted


def test_the_callback_totals_tokens_from_each_llm_response() -> None:
    counter = BudgetCallback()

    counter.on_llm_end(llm_result("claude-sonnet-4-6", 100, 50), run_id=uuid4())
    counter.on_llm_end(llm_result("claude-opus-4-8", 200, 80), run_id=uuid4())

    assert counter.total_tokens == 100 + 50 + 200 + 80


def test_the_callback_keeps_usage_per_model() -> None:
    counter = BudgetCallback()

    counter.on_llm_end(llm_result("claude-sonnet-4-6", 100, 50), run_id=uuid4())
    counter.on_llm_end(llm_result("claude-sonnet-4-6", 10, 5), run_id=uuid4())
    counter.on_llm_end(llm_result("claude-opus-4-8", 200, 80), run_id=uuid4())

    usage = counter.usage
    assert usage["claude-sonnet-4-6"].input_tokens == 110
    assert usage["claude-sonnet-4-6"].output_tokens == 55
    assert usage["claude-opus-4-8"].input_tokens == 200


def test_the_callback_captures_cache_read_tokens_per_model() -> None:
    counter = BudgetCallback()

    counter.on_llm_end(llm_result("claude-sonnet-4-6", 1_000, 50, cache_read=800), run_id=uuid4())
    counter.on_llm_end(llm_result("claude-sonnet-4-6", 1_000, 50, cache_read=900), run_id=uuid4())

    usage = counter.usage
    assert usage["claude-sonnet-4-6"].input_tokens == 2_000
    assert usage["claude-sonnet-4-6"].cached_input_tokens == 1_700


def test_a_response_without_a_cache_field_reports_no_cached_tokens() -> None:
    counter = BudgetCallback()

    counter.on_llm_end(llm_result("gpt-5.5", 1_000, 50), run_id=uuid4())

    assert counter.usage["gpt-5.5"].cached_input_tokens == 0


def test_the_callback_ignores_responses_without_token_usage() -> None:
    counter = BudgetCallback()
    message = AIMessage(content="no usage here")
    counter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=uuid4())

    assert counter.total_tokens == 0
    assert counter.usage == {}


def test_the_default_budget_does_not_wrap_a_typical_interview() -> None:
    budget = default_session_budget(DEFAULT_TOKEN_BUDGET)
    typical_tokens = sum(expected_session_tokens(TYPICAL_TURNS))

    assert not budget.record(typical_tokens).exhausted


def test_the_default_reserve_covers_the_feedback_pass() -> None:
    budget = default_session_budget(DEFAULT_TOKEN_BUDGET)

    assert budget.reserve >= FEEDBACK_INPUT_TOKENS + FEEDBACK_OUTPUT_TOKENS


def test_the_default_reserve_is_the_configured_fraction_of_the_budget() -> None:
    budget = default_session_budget(60_000)

    assert budget.reserve == int(60_000 * RESERVE_FRACTION)

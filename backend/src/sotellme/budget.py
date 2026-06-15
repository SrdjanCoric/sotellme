from dataclasses import dataclass, replace
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from sotellme.pricing import ModelUsage

WRAP_FRACTION = 0.85
RESERVE_FRACTION = 0.15

# A safety cap, not a target: a typical interview (~270k tokens by the estimate in
# pricing.py) clears it with headroom, while a full-length one (~400k, the question cap)
# may wrap on budget first, which is the cost backstop working as intended. Calibrated
# 2026-06-15 against a real ~212k-token Gemini run; override with SOTELLME_TOKEN_BUDGET.
DEFAULT_TOKEN_BUDGET = 400_000


@dataclass(frozen=True)
class SessionBudget:
    total_budget: int
    reserve: int
    tokens_used: int = 0
    wrap_fraction: float = WRAP_FRACTION

    @property
    def interview_allowance(self) -> int:
        return max(self.total_budget - self.reserve, 0)

    @property
    def wrap_threshold(self) -> int:
        return int(self.wrap_fraction * self.interview_allowance)

    @property
    def exhausted(self) -> bool:
        return self.tokens_used >= self.wrap_threshold

    def record(self, tokens: int) -> "SessionBudget":
        return replace(self, tokens_used=self.tokens_used + tokens)


def default_session_budget(total_budget: int, tokens_used: int = 0) -> SessionBudget:
    return SessionBudget(
        total_budget=total_budget,
        reserve=int(total_budget * RESERVE_FRACTION),
        tokens_used=tokens_used,
    )


class BudgetCallback(BaseCallbackHandler):
    """Accumulates token usage per model across a session's LLM calls."""

    def __init__(self) -> None:
        self._usage: dict[str, list[int]] = {}

    def record(
        self, model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0
    ) -> None:
        bucket = self._usage.setdefault(model, [0, 0, 0])
        bucket[0] += input_tokens
        bucket[1] += output_tokens
        bucket[2] += cached_input_tokens

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if not usage:
                    continue
                metadata = getattr(message, "response_metadata", {}) or {}
                model = metadata.get("model_name") or metadata.get("model") or "unknown"
                cache_read = (usage.get("input_token_details") or {}).get("cache_read", 0)
                self.record(
                    model,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    cache_read,
                )

    @property
    def total_tokens(self) -> int:
        return sum(used_input + used_output for used_input, used_output, _ in self._usage.values())

    @property
    def usage(self) -> dict[str, ModelUsage]:
        return {
            model: ModelUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            )
            for model, (input_tokens, output_tokens, cached_input_tokens) in self._usage.items()
        }

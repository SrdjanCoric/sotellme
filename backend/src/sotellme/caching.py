from collections.abc import Sequence

from langchain_core.messages import SystemMessage

ANTHROPIC_PROVIDER = "anthropic"
EPHEMERAL_CACHE_CONTROL = {"type": "ephemeral"}

# Anthropic ignores a cache breakpoint on a block below its per-model minimum (~1024 tokens for
# Sonnet/Opus, more for Haiku). Gate on a conservative character proxy (~4 chars/token) so only a
# system prompt actually large enough to cache carries a breakpoint.
MIN_CACHEABLE_CHARS = 4096

PromptMessage = tuple[str, str]
CachedMessages = list[PromptMessage | SystemMessage]


def cache_system_prompt(messages: Sequence[PromptMessage], provider: str) -> CachedMessages:
    out: CachedMessages = list(messages)
    if provider != ANTHROPIC_PROVIDER or not out:
        return out
    role, text = messages[0]
    if role != "system" or len(text) < MIN_CACHEABLE_CHARS:
        return out
    out[0] = SystemMessage(
        content=[{"type": "text", "text": text, "cache_control": EPHEMERAL_CACHE_CONTROL}]
    )
    return out

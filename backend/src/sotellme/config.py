from collections.abc import Mapping
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

PROVIDER_DEFAULTS = {
    "anthropic": ("claude-sonnet-4-6", "claude-opus-4-8"),
    "openai": ("gpt-5.4-mini", "gpt-5.5"),
    "google_genai": ("gemini-3.5-flash", "gemini-3.1-pro-preview"),
}

PROVIDER_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}


class ModelConfigError(Exception):
    pass


class ModelConfig(BaseModel):
    provider: str
    fast_model: str
    smart_model: str


def resolve_model_config(
    env: Mapping[str, str],
    provider: str | None = None,
    fast_model: str | None = None,
    smart_model: str | None = None,
) -> ModelConfig:
    provider = provider or env.get("SOTELLME_PROVIDER")
    if not provider:
        raise ModelConfigError("No provider selected: pass --provider or set SOTELLME_PROVIDER.")
    if provider not in PROVIDER_DEFAULTS:
        valid = ", ".join(sorted(PROVIDER_DEFAULTS))
        raise ModelConfigError(f"Unknown provider {provider!r}: choose one of {valid}.")
    key_var = PROVIDER_KEY_VARS[provider]
    if not env.get(key_var):
        raise ModelConfigError(
            f"No API key found for {provider}: set the {key_var} environment variable."
        )
    default_fast, default_smart = PROVIDER_DEFAULTS[provider]
    return ModelConfig(
        provider=provider,
        fast_model=fast_model or env.get("SOTELLME_FAST_MODEL") or default_fast,
        smart_model=smart_model or env.get("SOTELLME_SMART_MODEL") or default_smart,
    )


def build_chat_model(config: ModelConfig, slot: Literal["fast", "smart"]) -> BaseChatModel:
    model_name = config.fast_model if slot == "fast" else config.smart_model
    return init_chat_model(model_name, model_provider=config.provider)

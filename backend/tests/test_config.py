import pytest

from sotellme.config import ModelConfigError, build_chat_model, resolve_model_config


def test_provider_arg_fills_both_slots_with_defaults() -> None:
    config = resolve_model_config(provider="anthropic", env={"ANTHROPIC_API_KEY": "sk-test"})

    assert config.provider == "anthropic"
    assert config.fast_model == "claude-sonnet-4-6"
    assert config.smart_model == "claude-opus-4-8"


def test_missing_api_key_fails_fast_naming_the_env_var() -> None:
    with pytest.raises(ModelConfigError, match="ANTHROPIC_API_KEY"):
        resolve_model_config(provider="anthropic", env={})


def test_provider_is_read_from_env_when_no_arg_given() -> None:
    config = resolve_model_config(
        env={"SOTELLME_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-test"}
    )

    assert config.provider == "anthropic"


def test_no_provider_anywhere_is_a_clear_error() -> None:
    with pytest.raises(ModelConfigError, match="SOTELLME_PROVIDER"):
        resolve_model_config(env={})


def test_build_chat_model_routes_slot_to_provider_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    config = resolve_model_config(provider="anthropic", env={"ANTHROPIC_API_KEY": "sk-test"})

    model = build_chat_model(config, "fast")

    assert "claude-sonnet-4-6" in str(getattr(model, "model", ""))


def test_openai_provider_defaults_and_key_var() -> None:
    config = resolve_model_config(provider="openai", env={"OPENAI_API_KEY": "sk-test"})

    assert config.fast_model == "gpt-5.4-mini"
    assert config.smart_model == "gpt-5.5"

    with pytest.raises(ModelConfigError, match="OPENAI_API_KEY"):
        resolve_model_config(provider="openai", env={})


def test_google_provider_defaults_and_key_var() -> None:
    config = resolve_model_config(provider="google_genai", env={"GOOGLE_API_KEY": "k"})

    assert config.fast_model == "gemini-3.5-flash"
    assert config.smart_model == "gemini-3.1-pro-preview"

    with pytest.raises(ModelConfigError, match="GOOGLE_API_KEY"):
        resolve_model_config(provider="google_genai", env={})


def test_slot_overrides_from_args_beat_env_and_defaults() -> None:
    config = resolve_model_config(
        provider="anthropic",
        fast_model="claude-sonnet-4-6",
        env={
            "ANTHROPIC_API_KEY": "sk-test",
            "SOTELLME_FAST_MODEL": "from-env",
            "SOTELLME_SMART_MODEL": "claude-opus-4-7",
        },
    )

    assert config.fast_model == "claude-sonnet-4-6"
    assert config.smart_model == "claude-opus-4-7"


def test_unknown_provider_lists_the_valid_choices() -> None:
    with pytest.raises(ModelConfigError, match="anthropic"):
        resolve_model_config(provider="copilot", env={})

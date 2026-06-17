import pytest

from sotellme.catalog import Catalog, default_catalog
from sotellme.config import (
    AGENT_TAG_PREFIX,
    AgentModel,
    ModelConfigError,
    build_chat_model,
    resolve_model_config,
)


def _with_agents(**agents: str) -> Catalog:
    return default_catalog().model_copy(update={"agents": agents})


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


def test_build_chat_model_tags_an_agent_model_with_its_role() -> None:
    config = resolve_model_config(provider="anthropic", env={"ANTHROPIC_API_KEY": "sk-test"})

    model = build_chat_model(config, "grader")

    assert f"{AGENT_TAG_PREFIX}grader" in (model.tags or [])


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


def test_every_agent_resolves_to_its_tier_default() -> None:
    config = resolve_model_config(provider="anthropic", env={"ANTHROPIC_API_KEY": "k"})

    assert config.agents["interviewer"] == AgentModel(
        provider="anthropic", model="claude-sonnet-4-6"
    )
    assert config.agents["researcher"].model == "claude-sonnet-4-6"
    assert config.agents["director"] == AgentModel(provider="anthropic", model="claude-opus-4-8")
    assert config.agents["grader"] == AgentModel(provider="anthropic", model="claude-opus-4-8")
    assert config.agents["coach"].model == "claude-opus-4-8"


def test_a_catalog_agents_entry_reassigns_only_that_agent() -> None:
    catalog = _with_agents(researcher="anthropic:claude-haiku-4-5-20251001")

    config = resolve_model_config(
        provider="anthropic", env={"ANTHROPIC_API_KEY": "k"}, catalog=catalog
    )

    assert config.agents["researcher"].model == "claude-haiku-4-5-20251001"
    assert config.agents["interviewer"].model == "claude-sonnet-4-6"


def test_an_agent_pinned_to_a_second_provider_needs_that_key() -> None:
    catalog = _with_agents(grader="openai:gpt-5.5")

    with pytest.raises(ModelConfigError, match="OPENAI_API_KEY"):
        resolve_model_config(provider="anthropic", env={"ANTHROPIC_API_KEY": "k"}, catalog=catalog)


def test_mixed_providers_resolve_when_both_keys_are_present() -> None:
    catalog = _with_agents(grader="openai:gpt-5.5")

    config = resolve_model_config(
        provider="anthropic",
        env={"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k2"},
        catalog=catalog,
    )

    assert config.agents["grader"] == AgentModel(provider="openai", model="gpt-5.5")
    assert config.agents["coach"].provider == "anthropic"


def test_a_catalog_agent_pinned_to_an_unlisted_model_is_rejected() -> None:
    catalog = _with_agents(grader="anthropic:ghost-model")

    with pytest.raises(ModelConfigError):
        resolve_model_config(provider="anthropic", env={"ANTHROPIC_API_KEY": "k"}, catalog=catalog)


def test_a_catalog_agent_with_an_unknown_role_is_rejected() -> None:
    catalog = _with_agents(butler="anthropic:claude-opus-4-8")

    with pytest.raises(ModelConfigError):
        resolve_model_config(provider="anthropic", env={"ANTHROPIC_API_KEY": "k"}, catalog=catalog)


def test_explicit_agent_overrides_take_top_precedence() -> None:
    catalog = _with_agents(grader="anthropic:claude-sonnet-4-6")

    config = resolve_model_config(
        provider="anthropic",
        env={"ANTHROPIC_API_KEY": "k"},
        catalog=catalog,
        agent_overrides={
            "grader": AgentModel(provider="anthropic", model="claude-haiku-4-5-20251001")
        },
    )

    assert config.agents["grader"].model == "claude-haiku-4-5-20251001"


def test_an_agent_override_with_an_unknown_provider_is_rejected() -> None:
    with pytest.raises(ModelConfigError, match="copilot"):
        resolve_model_config(
            provider="anthropic",
            env={"ANTHROPIC_API_KEY": "k"},
            agent_overrides={"grader": AgentModel(provider="copilot", model="ghost-model")},
        )


def test_an_agent_override_pinned_to_an_unlisted_model_is_rejected() -> None:
    with pytest.raises(ModelConfigError, match="ghost-model"):
        resolve_model_config(
            provider="anthropic",
            env={"ANTHROPIC_API_KEY": "k"},
            agent_overrides={"grader": AgentModel(provider="anthropic", model="ghost-model")},
        )


def test_build_chat_model_routes_an_agent_role(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    config = resolve_model_config(provider="anthropic", env={"ANTHROPIC_API_KEY": "sk-test"})

    model = build_chat_model(config, "grader")

    assert "claude-opus-4-8" in str(getattr(model, "model", ""))

from pathlib import Path

import pytest

from sotellme.catalog import CatalogError, default_catalog, load_catalog


def test_default_catalog_lists_the_three_providers_with_their_tier_defaults() -> None:
    catalog = default_catalog()

    assert set(catalog.providers) == {"anthropic", "openai", "google_genai"}
    assert catalog.providers["anthropic"].fast == "claude-sonnet-4-6"
    assert catalog.providers["anthropic"].smart == "claude-opus-4-8"


def test_default_catalog_has_no_agent_assignments() -> None:
    assert default_catalog().agents == {}


def test_loading_without_an_override_is_the_default(tmp_path: Path) -> None:
    assert load_catalog(tmp_path) == default_catalog()


def test_a_user_override_replaces_one_provider_and_leaves_the_rest(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text(
        "[providers.anthropic]\n"
        'fast = "claude-haiku-4-5-20251001"\n'
        'smart = "claude-opus-4-8"\n'
        'models = ["claude-opus-4-8", "claude-haiku-4-5-20251001"]\n'
    )

    catalog = load_catalog(tmp_path)

    assert catalog.providers["anthropic"].fast == "claude-haiku-4-5-20251001"
    assert catalog.providers["openai"].smart == "gpt-5.5"


def test_a_user_override_can_add_agent_assignments(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text('[agents]\nresearcher = "openai:gpt-5.4-mini"\n')

    assert load_catalog(tmp_path).agents == {"researcher": "openai:gpt-5.4-mini"}


def test_malformed_override_raises_a_catalog_error(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text("this = = not toml")

    with pytest.raises(CatalogError):
        load_catalog(tmp_path)


def test_a_tier_default_outside_the_models_list_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text(
        '[providers.openai]\nfast = "ghost"\nsmart = "gpt-5.5"\nmodels = ["gpt-5.5"]\n'
    )

    with pytest.raises(CatalogError):
        load_catalog(tmp_path)

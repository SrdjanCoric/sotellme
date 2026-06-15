from pathlib import Path

import pytest

from sotellme.catalog import CatalogError, default_catalog, load_catalog


def test_the_default_catalog_prices_the_recommended_anthropic_models() -> None:
    prices = default_catalog().prices

    assert prices["claude-opus-4-8"].input == 5.0
    assert prices["claude-opus-4-8"].output == 25.0
    assert prices["claude-sonnet-4-6"].input == 3.0
    assert prices["claude-sonnet-4-6"].output == 15.0


def test_the_default_catalog_prices_cached_input_below_the_full_input_rate() -> None:
    prices = default_catalog().prices

    # Anthropic reads cached input at 10% of the input rate.
    assert prices["claude-opus-4-8"].cached_input == 0.5
    # OpenAI at 50%, Gemini at 25%.
    assert prices["gpt-5.5"].cached_input == 2.5
    assert prices["gemini-3.1-pro-preview"].cached_input == 0.5
    for price in prices.values():
        assert price.cached_input is not None
        assert price.cached_input < price.input


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


def test_a_user_override_can_replace_a_models_price(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text(
        '[prices."claude-opus-4-8"]\ninput = 9.0\noutput = 40.0\n'
    )

    catalog = load_catalog(tmp_path)

    assert catalog.prices["claude-opus-4-8"].input == 9.0
    assert catalog.prices["claude-opus-4-8"].output == 40.0
    assert catalog.prices["claude-sonnet-4-6"].input == 3.0


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

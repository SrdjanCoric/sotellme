"""Load the model catalog defining providers, agent assignments, and prices."""

import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, model_validator

CATALOG_FILENAME = "models.toml"


class CatalogError(Exception):
    """Raised when the model catalog cannot be parsed or fails validation."""


class ProviderCatalog(BaseModel):
    """Catalog entry for a single provider."""

    fast: str
    smart: str
    models: list[str]

    @model_validator(mode="after")
    def _defaults_are_listed(self) -> "ProviderCatalog":
        """Validate that the fast and smart default models appear in the models list."""
        for tier, name in (("fast", self.fast), ("smart", self.smart)):
            if name not in self.models:
                raise ValueError(f"the {tier} default {name!r} is not in this provider's models")
        return self


class ModelPrice(BaseModel):
    """Per-token pricing for a model."""

    input: float
    output: float
    cached_input: float | None = None

    @property
    def cached_input_rate(self) -> float:
        """Return the cached input rate, falling back to the input rate when unset."""
        return self.input if self.cached_input is None else self.cached_input


class Catalog(BaseModel):
    """The full model catalog of providers, agent assignments, and prices."""

    providers: dict[str, ProviderCatalog] = {}
    agents: dict[str, str] = {}
    prices: dict[str, ModelPrice] = {}


def _parse(text: str, source: str) -> Catalog:
    """Parse TOML catalog text into a Catalog, raising CatalogError on failure."""
    try:
        data: dict[str, Any] = tomllib.loads(text)
        return Catalog(
            providers=data.get("providers", {}),
            agents=data.get("agents", {}),
            prices=data.get("prices", {}),
        )
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise CatalogError(f"Could not read the model catalog at {source}: {exc}") from exc


def default_catalog() -> Catalog:
    """Load the catalog packaged with the sotellme distribution.

    Returns:
        The parsed packaged catalog.

    Raises:
        CatalogError: If the packaged catalog cannot be parsed or validated.
    """
    text = resources.files("sotellme").joinpath(CATALOG_FILENAME).read_text()
    return _parse(text, f"the packaged {CATALOG_FILENAME}")


def load_catalog(data_dir: Path) -> Catalog:
    """Load the catalog, merging any user override on top of the packaged defaults.

    Args:
        data_dir: Directory that may contain a user catalog override file.

    Returns:
        The packaged catalog, with any user override merged in per top-level section.

    Raises:
        CatalogError: If the user override file cannot be parsed or validated.
    """
    catalog = default_catalog()
    override = data_dir / CATALOG_FILENAME
    if not override.exists():
        return catalog
    user = _parse(override.read_text(), str(override))
    return Catalog(
        providers={**catalog.providers, **user.providers},
        agents={**catalog.agents, **user.agents},
        prices={**catalog.prices, **user.prices},
    )

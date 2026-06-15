import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, model_validator

CATALOG_FILENAME = "models.toml"


class CatalogError(Exception):
    pass


class ProviderCatalog(BaseModel):
    fast: str
    smart: str
    models: list[str]

    @model_validator(mode="after")
    def _defaults_are_listed(self) -> "ProviderCatalog":
        for tier, name in (("fast", self.fast), ("smart", self.smart)):
            if name not in self.models:
                raise ValueError(f"the {tier} default {name!r} is not in this provider's models")
        return self


class Catalog(BaseModel):
    providers: dict[str, ProviderCatalog] = {}
    agents: dict[str, str] = {}


def _parse(text: str, source: str) -> Catalog:
    try:
        data: dict[str, Any] = tomllib.loads(text)
        return Catalog(providers=data.get("providers", {}), agents=data.get("agents", {}))
    except (tomllib.TOMLDecodeError, ValidationError) as exc:
        raise CatalogError(f"Could not read the model catalog at {source}: {exc}") from exc


def default_catalog() -> Catalog:
    text = resources.files("sotellme").joinpath(CATALOG_FILENAME).read_text()
    return _parse(text, f"the packaged {CATALOG_FILENAME}")


def load_catalog(data_dir: Path) -> Catalog:
    catalog = default_catalog()
    override = data_dir / CATALOG_FILENAME
    if not override.exists():
        return catalog
    user = _parse(override.read_text(), str(override))
    return Catalog(
        providers={**catalog.providers, **user.providers},
        agents={**catalog.agents, **user.agents},
    )

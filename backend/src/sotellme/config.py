from collections.abc import Mapping

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from sotellme.catalog import Catalog, default_catalog

PROVIDER_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}

AGENT_TIERS: dict[str, str] = {
    "parser": "fast",
    "researcher": "fast",
    "role_builder": "fast",
    "director": "fast",
    "interviewer": "fast",
    "assessor": "fast",
    "guardrail": "fast",
    "grader": "smart",
    "coach": "smart",
}

AGENT_ROLES = tuple(AGENT_TIERS)

AGENT_TAG_PREFIX = "sotellme-agent:"


class ModelConfigError(Exception):
    pass


class AgentModel(BaseModel):
    provider: str
    model: str


class ModelConfig(BaseModel):
    provider: str
    fast_model: str
    smart_model: str
    agents: dict[str, AgentModel]


def _agent_from_spec(role: str, spec: str, catalog: Catalog) -> AgentModel:
    provider, _, model = spec.partition(":")
    if not _ or not model:
        raise ModelConfigError(
            f"Agent {role!r} must be assigned as 'provider:model', not {spec!r}."
        )
    if provider not in catalog.providers:
        valid = ", ".join(sorted(catalog.providers))
        raise ModelConfigError(f"Agent {role!r} names unknown provider {provider!r}: {valid}.")
    if model not in catalog.providers[provider].models:
        raise ModelConfigError(
            f"Agent {role!r} names {model!r}, which is not in the {provider} catalog."
        )
    return AgentModel(provider=provider, model=model)


def _require_keys(agents: Mapping[str, AgentModel], env: Mapping[str, str]) -> None:
    missing = {
        PROVIDER_KEY_VARS[agent.provider]
        for agent in agents.values()
        if not env.get(PROVIDER_KEY_VARS[agent.provider])
    }
    if missing:
        names = ", ".join(sorted(missing))
        raise ModelConfigError(
            f"No API key found for the selected models: set the {names} environment variable(s)."
        )


def resolve_model_config(
    env: Mapping[str, str],
    provider: str | None = None,
    fast_model: str | None = None,
    smart_model: str | None = None,
    catalog: Catalog | None = None,
    agent_overrides: Mapping[str, AgentModel] | None = None,
) -> ModelConfig:
    catalog = catalog or default_catalog()
    provider = provider or env.get("SOTELLME_PROVIDER")
    if not provider:
        raise ModelConfigError("No provider selected: pass --provider or set SOTELLME_PROVIDER.")
    if provider not in catalog.providers:
        valid = ", ".join(sorted(catalog.providers))
        raise ModelConfigError(f"Unknown provider {provider!r}: choose one of {valid}.")

    defaults = catalog.providers[provider]
    fast = fast_model or env.get("SOTELLME_FAST_MODEL") or defaults.fast
    smart = smart_model or env.get("SOTELLME_SMART_MODEL") or defaults.smart

    agents = {
        role: AgentModel(provider=provider, model=fast if tier == "fast" else smart)
        for role, tier in AGENT_TIERS.items()
    }
    for role, spec in catalog.agents.items():
        if role not in AGENT_TIERS:
            valid = ", ".join(AGENT_ROLES)
            raise ModelConfigError(f"Unknown agent {role!r} in the catalog: choose one of {valid}.")
        agents[role] = _agent_from_spec(role, spec, catalog)
    for role, override in (agent_overrides or {}).items():
        agents[role] = override

    _require_keys(agents, env)
    return ModelConfig(provider=provider, fast_model=fast, smart_model=smart, agents=agents)


def build_chat_model(config: ModelConfig, key: str) -> BaseChatModel:
    if key in config.agents:
        agent = config.agents[key]
        return init_chat_model(
            agent.model, model_provider=agent.provider, tags=[f"{AGENT_TAG_PREFIX}{key}"]
        )
    if key in ("fast", "smart"):
        model = config.fast_model if key == "fast" else config.smart_model
        return init_chat_model(model, model_provider=config.provider)
    raise ModelConfigError(f"Unknown model key {key!r}.")

"""Optional Langfuse tracing wiring driven by environment configuration."""

from collections.abc import Mapping
from importlib.util import find_spec

from langchain_core.callbacks import BaseCallbackHandler


class TracingError(Exception):
    """Raised when tracing is enabled but the Langfuse client is not installed."""

    pass


def langfuse_configured(env: Mapping[str, str]) -> bool:
    """Report whether Langfuse credentials are present in the environment."""
    return bool(env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY"))


def langfuse_callbacks(env: Mapping[str, str]) -> list[BaseCallbackHandler]:
    """Build the Langfuse callback handlers when tracing is configured.

    Args:
        env: Environment variable mapping.

    Returns:
        A list with a single Langfuse callback handler when configured, or an empty list when
        tracing is not configured.

    Raises:
        TracingError: If tracing is configured but the langfuse package is not installed.
    """
    if not langfuse_configured(env):
        return []
    if find_spec("langfuse") is None:
        raise TracingError(
            "Langfuse tracing is on (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set), "
            "but the langfuse client isn't installed in this environment. Relaunch with the "
            'tracing extra included — uvx --from "sotellme[web,tracing]" sotellme web — and set '
            "LANGFUSE_HOST to point at your local or Docker Langfuse. To run without tracing, "
            "unset those variables."
        )
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]

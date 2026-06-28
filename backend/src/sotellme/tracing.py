"""Optional Langfuse tracing wiring driven by environment configuration."""

import os
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
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


def trace_session(
    session_id: str,
    user_id: str | None = None,
    tags: list[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> AbstractContextManager[object]:
    """Group every span of one interview under a single Langfuse session.

    Returns Langfuse's ``propagate_attributes`` context manager — which stamps the
    ``session_id`` (and optional ``user_id``/``tags``) onto the active trace and all of
    its child spans, so the turns of one interview roll up into one session — when tracing
    is both configured (its env vars are set) and the optional ``langfuse`` client is
    installed, and a no-op context manager otherwise. The single configuration gate keeps
    tracing off by default and the engine free of any hard dependency on the tracing extra.

    Args:
        session_id: Canonical session id to group on — the interview's LangGraph thread id.
        user_id: Optional caller-supplied identity; omitted from the trace when None.
        tags: Optional trace tags (company/role/level), with unset values already dropped.
        env: Environment mapping to read the Langfuse configuration from; defaults to the
            process environment.

    Returns:
        A ``propagate_attributes`` context manager when tracing is available, else a no-op.
    """
    resolved_env = os.environ if env is None else env
    if not langfuse_configured(resolved_env) or find_spec("langfuse") is None:
        return nullcontext()
    from langfuse import propagate_attributes

    # Bind to a typed local: without the tracing extra installed, langfuse is untyped (Any),
    # and returning Any directly trips mypy's --strict no-any-return in that CI configuration.
    propagation: AbstractContextManager[object] = propagate_attributes(
        session_id=session_id, user_id=user_id, tags=tags
    )
    return propagation

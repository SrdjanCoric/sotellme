from collections.abc import Mapping
from importlib.util import find_spec

from langchain_core.callbacks import BaseCallbackHandler


class TracingError(Exception):
    pass


def langfuse_configured(env: Mapping[str, str]) -> bool:
    return bool(env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY"))


def langfuse_callbacks(env: Mapping[str, str]) -> list[BaseCallbackHandler]:
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

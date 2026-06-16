from collections.abc import Mapping
from importlib.util import find_spec

from langchain_core.callbacks import BaseCallbackHandler


class TracingError(Exception):
    pass


def langfuse_callbacks(env: Mapping[str, str]) -> list[BaseCallbackHandler]:
    if not (env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY")):
        return []
    if find_spec("langfuse") is None:
        raise TracingError(
            "Langfuse tracing is on (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set), "
            "but the langfuse package isn't installed. Install the tracing extra "
            "(sotellme[tracing]), or unset those variables to run without tracing."
        )
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]

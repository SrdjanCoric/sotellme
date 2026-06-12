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
            "Langfuse env vars are set but the langfuse package is not installed. "
            "Install it with: pip install 'sotellme[tracing]'"
        )
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]

import sys

import pytest

import sotellme.tracing
from sotellme.tracing import TracingError, langfuse_callbacks


def test_absent_env_vars_mean_no_callbacks_and_no_import() -> None:
    assert langfuse_callbacks(env={}) == []
    assert "langfuse" not in sys.modules


def test_partial_env_vars_mean_no_callbacks() -> None:
    assert langfuse_callbacks(env={"LANGFUSE_PUBLIC_KEY": "pk"}) == []


def test_configured_but_uninstalled_langfuse_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sotellme.tracing, "find_spec", lambda name: None)

    with pytest.raises(TracingError) as exc_info:
        langfuse_callbacks(env={"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"})

    message = str(exc_info.value)
    assert "sotellme[tracing]" in message
    assert "without tracing" in message

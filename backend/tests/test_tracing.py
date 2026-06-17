import sys

import pytest

import sotellme.tracing
from sotellme.tracing import TracingError, langfuse_callbacks, langfuse_configured


def test_absent_env_vars_mean_no_callbacks_and_no_import() -> None:
    assert langfuse_callbacks(env={}) == []
    assert "langfuse" not in sys.modules


def test_partial_env_vars_mean_no_callbacks() -> None:
    assert langfuse_callbacks(env={"LANGFUSE_PUBLIC_KEY": "pk"}) == []


def test_langfuse_is_unconfigured_by_default() -> None:
    assert langfuse_configured(env={}) is False
    assert langfuse_configured(env={"LANGFUSE_PUBLIC_KEY": "pk"}) is False


def test_langfuse_is_configured_only_when_both_keys_are_set() -> None:
    assert langfuse_configured(env={"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"})


def test_configured_but_uninstalled_langfuse_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sotellme.tracing, "find_spec", lambda name: None)

    with pytest.raises(TracingError) as exc_info:
        langfuse_callbacks(env={"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"})

    message = str(exc_info.value)
    assert 'uvx --from "sotellme[web,tracing]" sotellme web' in message
    assert "LANGFUSE_HOST" in message
    assert "without tracing" in message

import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

import sotellme.tracing
from sotellme.tracing import (
    TracingError,
    langfuse_callbacks,
    langfuse_configured,
    trace_session,
)


def test_absent_env_vars_mean_no_callbacks_and_no_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Scope the no-import check to this call: other engine paths legitimately import langfuse
    # when tracing is configured and the extra is installed, so the global module table can't
    # be the witness on its own.
    monkeypatch.delitem(sys.modules, "langfuse", raising=False)
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


def test_trace_session_is_a_noop_when_langfuse_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "langfuse", raising=False)
    monkeypatch.setattr(sotellme.tracing, "find_spec", lambda name: None)

    # configured, but the optional client isn't installed — the helper must still no-op,
    # never import, and keep the engine free of any hard dependency on the tracing extra
    with trace_session(
        "thread-abc",
        user_id="acct-7",
        tags=["acme"],
        env={"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"},
    ):
        pass

    assert "langfuse" not in sys.modules


def test_trace_session_propagates_session_attributes_when_configured_and_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    @contextmanager
    def fake_propagate_attributes(
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        **_: object,
    ) -> Iterator[None]:
        recorded.update(session_id=session_id, user_id=user_id, tags=tags, entered=True)
        yield
        recorded["exited"] = True

    monkeypatch.setitem(
        sys.modules,
        "langfuse",
        types.SimpleNamespace(propagate_attributes=fake_propagate_attributes),
    )
    monkeypatch.setattr(sotellme.tracing, "find_spec", lambda name: object())

    with trace_session(
        "thread-abc",
        user_id="acct-7",
        tags=["Acme", "Backend Engineer", "senior"],
        env={"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"},
    ):
        assert recorded["entered"] is True

    assert recorded["session_id"] == "thread-abc"
    assert recorded["user_id"] == "acct-7"
    assert recorded["tags"] == ["Acme", "Backend Engineer", "senior"]
    assert recorded["exited"] is True


def test_trace_session_is_a_noop_when_tracing_is_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = False

    @contextmanager
    def fake_propagate_attributes(**_: object) -> Iterator[None]:
        nonlocal entered
        entered = True
        yield

    monkeypatch.setitem(
        sys.modules,
        "langfuse",
        types.SimpleNamespace(propagate_attributes=fake_propagate_attributes),
    )
    monkeypatch.setattr(sotellme.tracing, "find_spec", lambda name: object())

    # langfuse is importable, but no credentials are set — tracing stays off by default
    with trace_session("thread-abc", env={}):
        pass

    assert entered is False

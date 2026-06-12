import inspect
from pathlib import Path

import pytest

import sotellme.prompts
from sotellme.config import PROVIDER_KEY_VARS
from sotellme.engine import InterviewEngine
from sotellme.profile import CandidateProfile, Role

SENTINEL = "SECRET-SENTINEL-do-not-leak"


def stub_parser(cv_text: str) -> CandidateProfile:
    return CandidateProfile(
        roles=[Role(title="Senior Engineer", organization="Acme")],
        projects=[],
        quantified_claims=[],
        technologies=[],
    )


def test_no_env_secret_reaches_session_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key_var in PROVIDER_KEY_VARS.values():
        monkeypatch.setenv(key_var, f"{SENTINEL}-{key_var}")

    cv = tmp_path / "cv.md"
    cv.write_text("# Jane Doe\nSenior Engineer at Acme")
    with InterviewEngine(data_dir=tmp_path / "data", profile_parser=stub_parser) as engine:
        session = engine.start(cv)
        engine.submit_answer(session.thread_id, "An answer.")
        state = engine._graph.get_state({"configurable": {"thread_id": session.thread_id}})

    session_text = session.question + repr(state.values)
    assert SENTINEL not in session_text


def test_prompt_module_never_reads_the_environment() -> None:
    source = inspect.getsource(sotellme.prompts)

    assert "environ" not in source
    assert "getenv" not in source


def test_prompt_constants_contain_no_secret_material() -> None:
    prompt_strings = [
        value
        for name, value in vars(sotellme.prompts).items()
        if isinstance(value, str) and not name.startswith("__")
    ]

    assert prompt_strings, "expected at least one prompt artifact to scan"
    for text in prompt_strings:
        assert "API_KEY" not in text
        assert "sk-" not in text

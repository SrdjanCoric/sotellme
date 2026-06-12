from pathlib import Path

import pytest

from sotellme.engine import EngineError, InterviewEngine
from sotellme.prompts import FIXED_OPENING_QUESTION


def write_cv(tmp_path: Path) -> Path:
    cv = tmp_path / "cv.md"
    cv.write_text("# Jane Doe\nSenior Engineer at Acme")
    return cv


def test_start_returns_the_opening_question(tmp_path: Path) -> None:
    with InterviewEngine(data_dir=tmp_path / "data") as engine:
        session = engine.start(write_cv(tmp_path))

    assert session.question == FIXED_OPENING_QUESTION
    assert session.thread_id


def test_submitted_answer_round_trips_through_the_graph(tmp_path: Path) -> None:
    with InterviewEngine(data_dir=tmp_path / "data") as engine:
        session = engine.start(write_cv(tmp_path))
        recorded = engine.submit_answer(session.thread_id, "I led the Acme migration.")

    assert recorded == "I led the Acme migration."


def test_session_resumes_across_engine_instances(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with InterviewEngine(data_dir=data_dir) as engine:
        started = engine.start(write_cv(tmp_path))

    with InterviewEngine(data_dir=data_dir) as engine:
        resumed = engine.resume_latest()
        assert resumed.thread_id == started.thread_id
        assert resumed.question == FIXED_OPENING_QUESTION
        recorded = engine.submit_answer(resumed.thread_id, "Resumed answer.")

    assert recorded == "Resumed answer."


def test_resume_with_no_session_is_a_clear_error(tmp_path: Path) -> None:
    with (
        InterviewEngine(data_dir=tmp_path / "data") as engine,
        pytest.raises(EngineError, match="No session to resume"),
    ):
        engine.resume_latest()


def test_resume_after_finished_session_is_a_clear_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with InterviewEngine(data_dir=data_dir) as engine:
        session = engine.start(write_cv(tmp_path))
        engine.submit_answer(session.thread_id, "Done.")

        with pytest.raises(EngineError, match="already finished"):
            engine.resume_latest()

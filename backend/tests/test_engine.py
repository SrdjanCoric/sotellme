from pathlib import Path

import pytest

from sotellme.engine import EngineError, InterviewEngine
from sotellme.profile import CandidateProfile, Role
from sotellme.prompts import FIXED_OPENING_QUESTION

STUB_PROFILE = CandidateProfile(
    roles=[Role(title="Senior Engineer", organization="Acme")],
    projects=[],
    quantified_claims=["Led the Acme migration"],
    technologies=["Python"],
)


def stub_parser(cv_text: str) -> CandidateProfile:
    return STUB_PROFILE


def write_cv(tmp_path: Path) -> Path:
    cv = tmp_path / "cv.md"
    cv.write_text("# Jane Doe\nSenior Engineer at Acme")
    return cv


def test_start_returns_the_opening_question(tmp_path: Path) -> None:
    with InterviewEngine(data_dir=tmp_path / "data", profile_parser=stub_parser) as engine:
        session = engine.start(write_cv(tmp_path))

    assert session.question == FIXED_OPENING_QUESTION
    assert session.thread_id


def test_submitted_answer_round_trips_through_the_graph(tmp_path: Path) -> None:
    with InterviewEngine(data_dir=tmp_path / "data", profile_parser=stub_parser) as engine:
        session = engine.start(write_cv(tmp_path))
        recorded = engine.submit_answer(session.thread_id, "I led the Acme migration.")

    assert recorded == "I led the Acme migration."


def test_session_resumes_across_engine_instances(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with InterviewEngine(data_dir=data_dir, profile_parser=stub_parser) as engine:
        started = engine.start(write_cv(tmp_path))

    with InterviewEngine(data_dir=data_dir, profile_parser=stub_parser) as engine:
        resumed = engine.resume_latest()
        assert resumed.thread_id == started.thread_id
        assert resumed.question == FIXED_OPENING_QUESTION
        recorded = engine.submit_answer(resumed.thread_id, "Resumed answer.")

    assert recorded == "Resumed answer."


def test_resume_with_no_session_is_a_clear_error(tmp_path: Path) -> None:
    with (
        InterviewEngine(data_dir=tmp_path / "data", profile_parser=stub_parser) as engine,
        pytest.raises(EngineError, match="No session to resume"),
    ):
        engine.resume_latest()


def test_start_parses_the_extracted_cv_into_a_profile(tmp_path: Path) -> None:
    seen: list[str] = []

    def recording_parser(cv_text: str) -> CandidateProfile:
        seen.append(cv_text)
        return STUB_PROFILE

    with InterviewEngine(data_dir=tmp_path / "data", profile_parser=recording_parser) as engine:
        session = engine.start(write_cv(tmp_path))

    assert seen == ["# Jane Doe\nSenior Engineer at Acme"]
    assert session.profile == STUB_PROFILE


def test_profile_survives_resume_across_engine_instances(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with InterviewEngine(data_dir=data_dir, profile_parser=stub_parser) as engine:
        engine.start(write_cv(tmp_path))

    def failing_parser(cv_text: str) -> CandidateProfile:
        raise AssertionError("resume must not re-parse the CV")

    with InterviewEngine(data_dir=data_dir, profile_parser=failing_parser) as engine:
        resumed = engine.resume_latest()

    assert resumed.profile == STUB_PROFILE


def test_resume_after_finished_session_is_a_clear_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with InterviewEngine(data_dir=data_dir, profile_parser=stub_parser) as engine:
        session = engine.start(write_cv(tmp_path))
        engine.submit_answer(session.thread_id, "Done.")

        with pytest.raises(EngineError, match="already finished"):
            engine.resume_latest()

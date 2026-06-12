import logging
from collections.abc import Sequence
from pathlib import Path

import pytest

from sotellme.coverage import Gap, StarFlags
from sotellme.engine import EngineError, InterviewEngine
from sotellme.interviewer import Turn
from sotellme.profile import CandidateProfile, Role

STUB_PROFILE = CandidateProfile(
    roles=[Role(title="Engineer", organization="Acme")],
    projects=[],
    quantified_claims=["Led the Acme migration"],
    technologies=["Python"],
)

OPENING_QUESTION = "Tell me about the Acme migration you led."

CLOSING_TURN = "That covers it, thanks for walking me through the migration."

COMPLETE_ANSWER = "situation task action result quantified"


def stub_parser(cv_text: str) -> CandidateProfile:
    return STUB_PROFILE


def keyword_flagger(answer: str) -> StarFlags:
    return StarFlags(
        situation="situation" in answer,
        task="task" in answer,
        action="action" in answer,
        result="result" in answer,
        quantified_result="quantified" in answer,
    )


class StubInterviewer:
    def __init__(self) -> None:
        self.probed_gaps: list[tuple[Gap, ...]] = []

    def opening_question(self, profile: CandidateProfile) -> str:
        return OPENING_QUESTION

    def probe_question(
        self, profile: CandidateProfile, transcript: Sequence[Turn], gaps: tuple[Gap, ...]
    ) -> str:
        self.probed_gaps.append(gaps)
        return f"Follow-up {len(self.probed_gaps)}: tell me more about the {gaps[0]}."

    def closing_turn(self, transcript: Sequence[Turn]) -> str:
        return CLOSING_TURN


def build_engine(data_dir: Path, followup_cap: int = 3) -> InterviewEngine:
    return InterviewEngine(
        data_dir=data_dir,
        profile_parser=stub_parser,
        star_flagger=keyword_flagger,
        interviewer=StubInterviewer(),
        followup_cap=followup_cap,
    )


def write_cv(tmp_path: Path) -> Path:
    cv = tmp_path / "cv.md"
    cv.write_text("# Jane Doe\nEngineer at Acme")
    return cv


def test_start_returns_the_interviewer_opening_question(tmp_path: Path) -> None:
    with build_engine(tmp_path / "data") as engine:
        session = engine.start(write_cv(tmp_path))

    assert session.question == OPENING_QUESTION
    assert session.thread_id
    assert session.profile == STUB_PROFILE


def test_a_star_complete_answer_finishes_without_belaboring(tmp_path: Path) -> None:
    with build_engine(tmp_path / "data") as engine:
        session = engine.start(write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, COMPLETE_ANSWER)

    assert result.finished


def test_a_finished_session_ends_with_the_interviewer_closing_turn(tmp_path: Path) -> None:
    with build_engine(tmp_path / "data") as engine:
        session = engine.start(write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, COMPLETE_ANSWER)

    assert result.closing == CLOSING_TURN


def test_a_probe_turn_carries_no_closing(tmp_path: Path) -> None:
    with build_engine(tmp_path / "data") as engine:
        session = engine.start(write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, "situation task action")

    assert not result.finished
    assert result.closing is None


def test_an_incomplete_answer_gets_a_probe_at_the_flagged_gap(tmp_path: Path) -> None:
    interviewer = StubInterviewer()
    engine = InterviewEngine(
        data_dir=tmp_path / "data",
        profile_parser=stub_parser,
        star_flagger=keyword_flagger,
        interviewer=interviewer,
    )
    with engine:
        session = engine.start(write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, "situation task action")

    assert result.next_question == "Follow-up 1: tell me more about the result."
    assert interviewer.probed_gaps == [("result",)]


def run_scripted_session(engine: InterviewEngine, cv: Path, answers: list[str]) -> int:
    session = engine.start(cv)
    questions_asked = 1
    for answer in answers:
        result = engine.submit_answer(session.thread_id, answer)
        if result.finished:
            return questions_asked
        questions_asked += 1
    raise AssertionError("the scripted session never finished")


def test_a_weak_transcript_draws_more_questions_than_a_strong_one(tmp_path: Path) -> None:
    with build_engine(tmp_path / "strong") as engine:
        strong_count = run_scripted_session(engine, write_cv(tmp_path), [COMPLETE_ANSWER])

    weak_answers = ["situation", "situation task", "situation task action", COMPLETE_ANSWER]
    with build_engine(tmp_path / "weak") as engine:
        weak_count = run_scripted_session(engine, write_cv(tmp_path), weak_answers)

    assert strong_count == 1
    assert weak_count == 4


def test_followups_are_bounded_by_the_cap(tmp_path: Path) -> None:
    evasive_answers = ["situation"] * 10
    with build_engine(tmp_path / "data", followup_cap=3) as engine:
        count = run_scripted_session(engine, write_cv(tmp_path), evasive_answers)

    assert count == 1 + 3


def test_pause_mid_competency_resumes_from_the_same_probe(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with build_engine(data_dir) as engine:
        session = engine.start(write_cv(tmp_path))
        probe = engine.submit_answer(session.thread_id, "situation task action")

    with build_engine(data_dir) as engine:
        resumed = engine.resume_latest()
        assert resumed.thread_id == session.thread_id
        assert resumed.question == probe.next_question
        result = engine.submit_answer(resumed.thread_id, COMPLETE_ANSWER)

    assert result.finished


def test_checkpoint_roundtrip_deserializes_only_registered_types(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data_dir = tmp_path / "data"
    with caplog.at_level(logging.WARNING):
        with build_engine(data_dir) as engine:
            session = engine.start(write_cv(tmp_path))
            engine.submit_answer(session.thread_id, "situation task")

        with build_engine(data_dir) as engine:
            resumed = engine.resume_latest()
            engine.submit_answer(resumed.thread_id, COMPLETE_ANSWER)

    assert "unregistered type" not in caplog.text


def test_profile_survives_resume_without_reparsing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with build_engine(data_dir) as engine:
        engine.start(write_cv(tmp_path))

    def failing_parser(cv_text: str) -> CandidateProfile:
        raise AssertionError("resume must not re-parse the CV")

    engine = InterviewEngine(
        data_dir=data_dir,
        profile_parser=failing_parser,
        star_flagger=keyword_flagger,
        interviewer=StubInterviewer(),
    )
    with engine:
        resumed = engine.resume_latest()

    assert resumed.profile == STUB_PROFILE


def test_resume_with_no_session_is_a_clear_error(tmp_path: Path) -> None:
    with (
        build_engine(tmp_path / "data") as engine,
        pytest.raises(EngineError, match="No session to resume"),
    ):
        engine.resume_latest()


def test_resume_after_finished_session_is_a_clear_error(tmp_path: Path) -> None:
    with build_engine(tmp_path / "data") as engine:
        session = engine.start(write_cv(tmp_path))
        engine.submit_answer(session.thread_id, COMPLETE_ANSWER)

        with pytest.raises(EngineError, match="already finished"):
            engine.resume_latest()

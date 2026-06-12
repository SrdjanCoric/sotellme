import logging
from collections.abc import Sequence
from pathlib import Path

import pytest

from sotellme.coverage import Gap, MotivationTopic, StarFlags
from sotellme.engine import EngineError, InterviewEngine, RoleBuilder, SessionHandle
from sotellme.interviewer import Turn
from sotellme.profile import CandidateProfile, Role
from sotellme.role import CompetencyWeight, RoleContext, TargetLevel

STUB_PROFILE = CandidateProfile(
    roles=[Role(title="Engineer", organization="Acme")],
    projects=[],
    quantified_claims=["Led the Acme migration"],
    technologies=["Python"],
)

OPENING_QUESTION = "Tell me a story about ownership."

CLOSING_TURN = "That covers it, thanks for walking me through the migration."

COMPLETE_ANSWER = "situation task action result quantified"

POSTING = "Senior Backend Engineer at Acme. You will own the billing platform."


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


def acme_context(target_level: TargetLevel | None = "senior") -> RoleContext:
    return RoleContext(
        company="Acme",
        role_title="Senior Backend Engineer",
        competencies=[
            CompetencyWeight(name="ownership", weight=5),
            CompetencyWeight(name="conflict", weight=4),
            CompetencyWeight(name="failure", weight=2),
        ],
        target_level=target_level,
    )


def builder_returning(context: RoleContext) -> RoleBuilder:
    def build(posting_text: str) -> RoleContext:
        return context

    return build


class StubInterviewer:
    def __init__(self) -> None:
        self.probed_gaps: list[tuple[Gap, ...]] = []
        self.motivation_topics: list[MotivationTopic] = []

    def competency_question(
        self, profile: CandidateProfile, transcript: Sequence[Turn], competency: str
    ) -> str:
        return f"Tell me a story about {competency}."

    def probe_question(
        self, profile: CandidateProfile, transcript: Sequence[Turn], gaps: tuple[Gap, ...]
    ) -> str:
        self.probed_gaps.append(gaps)
        return f"Follow-up {len(self.probed_gaps)}: tell me more about the {gaps[0]}."

    def motivation_question(
        self,
        context: RoleContext,
        posting_text: str,
        transcript: Sequence[Turn],
        topic: MotivationTopic,
    ) -> str:
        self.motivation_topics.append(topic)
        return f"Why this {topic}?"

    def closing_turn(self, transcript: Sequence[Turn]) -> str:
        return CLOSING_TURN


def build_engine(
    data_dir: Path,
    followup_cap: int = 3,
    max_competencies: int = 1,
    role_builder: RoleBuilder | None = None,
    interviewer: StubInterviewer | None = None,
) -> InterviewEngine:
    return InterviewEngine(
        data_dir=data_dir,
        profile_parser=stub_parser,
        star_flagger=keyword_flagger,
        interviewer=interviewer or StubInterviewer(),
        role_builder=role_builder or builder_returning(acme_context()),
        followup_cap=followup_cap,
        max_competencies=max_competencies,
    )


def write_cv(tmp_path: Path) -> Path:
    cv = tmp_path / "cv.md"
    cv.write_text("# Jane Doe\nEngineer at Acme")
    return cv


def start_past_setup(
    engine: InterviewEngine, cv: Path, posting_text: str | None = None
) -> SessionHandle:
    session = engine.start(cv, posting_text=posting_text)
    if session.needs_level:
        session = engine.submit_level(session.thread_id, "mid")
    return session


def test_start_returns_the_first_competency_question(tmp_path: Path) -> None:
    with build_engine(tmp_path / "data") as engine:
        session = start_past_setup(engine, write_cv(tmp_path))

    assert session.question == OPENING_QUESTION
    assert session.thread_id
    assert session.profile == STUB_PROFILE


def test_without_a_posting_the_level_is_asked_never_defaulted(tmp_path: Path) -> None:
    with build_engine(tmp_path / "data") as engine:
        session = engine.start(write_cv(tmp_path))

        assert session.needs_level
        assert session.question is None

        session = engine.submit_level(session.thread_id, "mid")

    assert not session.needs_level
    assert session.question == OPENING_QUESTION


def test_a_deduced_level_skips_the_setup_question(tmp_path: Path) -> None:
    builder = builder_returning(acme_context(target_level="senior"))
    with build_engine(tmp_path / "data", role_builder=builder) as engine:
        session = engine.start(write_cv(tmp_path), posting_text=POSTING)

    assert not session.needs_level
    assert session.question == OPENING_QUESTION


def test_a_posting_without_a_clear_level_pauses_to_ask(tmp_path: Path) -> None:
    builder = builder_returning(acme_context(target_level=None))
    with build_engine(tmp_path / "data", role_builder=builder) as engine:
        session = engine.start(write_cv(tmp_path), posting_text=POSTING)

        assert session.needs_level

        session = engine.submit_level(session.thread_id, "senior")

    assert session.question == OPENING_QUESTION


def test_the_level_ask_survives_a_restart(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with build_engine(data_dir) as engine:
        engine.start(write_cv(tmp_path))

    with build_engine(data_dir) as engine:
        resumed = engine.resume_latest()
        assert resumed.needs_level
        session = engine.submit_level(resumed.thread_id, "junior")

    assert session.question == OPENING_QUESTION


def test_a_session_walks_competencies_by_weight_then_motivation(tmp_path: Path) -> None:
    interviewer = StubInterviewer()
    builder = builder_returning(acme_context())
    engine = build_engine(
        tmp_path / "data", max_competencies=2, role_builder=builder, interviewer=interviewer
    )
    with engine:
        session = start_past_setup(engine, write_cv(tmp_path), posting_text=POSTING)
        assert session.question == "Tell me a story about ownership."

        second = engine.submit_answer(session.thread_id, COMPLETE_ANSWER)
        assert second.next_question == "Tell me a story about conflict."

        motivation_company = engine.submit_answer(session.thread_id, COMPLETE_ANSWER)
        assert motivation_company.next_question == "Why this company?"

        motivation_role = engine.submit_answer(session.thread_id, "The mission speaks to me.")
        assert motivation_role.next_question == "Why this role?"

        result = engine.submit_answer(session.thread_id, "The work itself.")

    assert result.finished
    assert result.closing == CLOSING_TURN
    assert interviewer.motivation_topics == ["company", "role"]


def test_motivation_answers_are_never_probed(tmp_path: Path) -> None:
    interviewer = StubInterviewer()
    engine = build_engine(tmp_path / "data", interviewer=interviewer)
    with engine:
        session = start_past_setup(engine, write_cv(tmp_path), posting_text=POSTING)
        engine.submit_answer(session.thread_id, COMPLETE_ANSWER)
        result = engine.submit_answer(session.thread_id, "vague motivation with no star at all")

    assert result.next_question == "Why this role?"
    assert interviewer.probed_gaps == []


def test_without_a_posting_there_is_no_motivation_segment(tmp_path: Path) -> None:
    interviewer = StubInterviewer()
    engine = build_engine(tmp_path / "data", max_competencies=2, interviewer=interviewer)
    with engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        engine.submit_answer(session.thread_id, COMPLETE_ANSWER)
        result = engine.submit_answer(session.thread_id, COMPLETE_ANSWER)

    assert result.finished
    assert interviewer.motivation_topics == []


def test_a_star_complete_answer_finishes_without_belaboring(tmp_path: Path) -> None:
    with build_engine(tmp_path / "data") as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, COMPLETE_ANSWER)

    assert result.finished


def test_a_finished_session_ends_with_the_interviewer_closing_turn(tmp_path: Path) -> None:
    with build_engine(tmp_path / "data") as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, COMPLETE_ANSWER)

    assert result.closing == CLOSING_TURN


def test_a_probe_turn_carries_no_closing(tmp_path: Path) -> None:
    with build_engine(tmp_path / "data") as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, "situation task action")

    assert not result.finished
    assert result.closing is None


def test_an_incomplete_answer_gets_a_probe_at_the_flagged_gap(tmp_path: Path) -> None:
    interviewer = StubInterviewer()
    engine = build_engine(tmp_path / "data", interviewer=interviewer)
    with engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, "situation task action")

    assert result.next_question == "Follow-up 1: tell me more about the result."
    assert interviewer.probed_gaps == [("result",)]


def test_the_followup_count_resets_for_each_competency(tmp_path: Path) -> None:
    interviewer = StubInterviewer()
    engine = build_engine(
        tmp_path / "data", followup_cap=1, max_competencies=2, interviewer=interviewer
    )
    with engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        probe = engine.submit_answer(session.thread_id, "situation")
        assert probe.next_question == "Follow-up 1: tell me more about the task."

        advance = engine.submit_answer(session.thread_id, "situation")
        assert advance.next_question == "Tell me a story about impact."

        probe_again = engine.submit_answer(session.thread_id, "situation")

    assert probe_again.next_question == "Follow-up 2: tell me more about the task."


def run_scripted_session(engine: InterviewEngine, cv: Path, answers: list[str]) -> int:
    session = start_past_setup(engine, cv)
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


def test_an_evasive_full_session_still_terminates(tmp_path: Path) -> None:
    evasive_answers = ["situation"] * 30
    engine = build_engine(tmp_path / "data", followup_cap=3, max_competencies=3)
    with engine:
        session = start_past_setup(engine, write_cv(tmp_path), posting_text=POSTING)
        questions = 1
        for answer in evasive_answers:
            result = engine.submit_answer(session.thread_id, answer)
            if result.finished:
                break
            questions += 1
        else:
            raise AssertionError("the evasive session never finished")

    assert questions == 3 * (1 + 3) + 2


def test_pause_mid_competency_resumes_from_the_same_probe(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with build_engine(data_dir) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
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
            session = start_past_setup(engine, write_cv(tmp_path), posting_text=POSTING)
            engine.submit_answer(session.thread_id, "situation task")

        with build_engine(data_dir) as engine:
            resumed = engine.resume_latest()
            engine.submit_answer(resumed.thread_id, COMPLETE_ANSWER)

    assert "unregistered type" not in caplog.text


def test_profile_survives_resume_without_reparsing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with build_engine(data_dir) as engine:
        start_past_setup(engine, write_cv(tmp_path))

    def failing_parser(cv_text: str) -> CandidateProfile:
        raise AssertionError("resume must not re-parse the CV")

    engine = InterviewEngine(
        data_dir=data_dir,
        profile_parser=failing_parser,
        star_flagger=keyword_flagger,
        interviewer=StubInterviewer(),
        role_builder=builder_returning(acme_context()),
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
        session = start_past_setup(engine, write_cv(tmp_path))
        engine.submit_answer(session.thread_id, COMPLETE_ANSWER)

        with pytest.raises(EngineError, match="already finished"):
            engine.resume_latest()

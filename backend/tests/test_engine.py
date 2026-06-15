import logging
from collections.abc import Sequence
from pathlib import Path

import pytest

from sotellme.assessor import AnswerAssessment, StarFlags
from sotellme.coach import AnswerAdvice, CoachReport, Drill
from sotellme.director import DirectorDecision, DirectorSituation
from sotellme.engine import Director, EngineError, InterviewEngine, RoleBuilder, SessionSnapshot
from sotellme.grader import AnswerScore, SessionGrade
from sotellme.interviewer import Turn
from sotellme.profile import CandidateProfile, Role
from sotellme.role import CompetencyWeight, RoleContext, TargetLevel

STUB_PROFILE = CandidateProfile(
    roles=[Role(title="Engineer", organization="Acme")],
    projects=[],
    quantified_claims=["Led the Acme migration"],
    technologies=["Python"],
)

OPENING_DECISION = DirectorDecision(
    action="new_topic", subject="their background", reason="the opener"
)

FOLLOW_UP_DECISION = DirectorDecision(
    action="follow_up", subject="the migration claim", reason="impact left unexplained"
)

WRAP_UP_DECISION = DirectorDecision(action="wrap_up", reason="enough signal")

CLOSING_TURN = "That covers it, thanks for walking me through the migration."

POSTING = "Senior Backend Engineer at Acme. You will own the billing platform."

BRIEF = "Acme builds billing software for veterinary clinics."


def stub_parser(cv_text: str) -> CandidateProfile:
    return STUB_PROFILE


def keyword_assessor(topic: str, transcript: Sequence[Turn]) -> AnswerAssessment:
    answer = transcript[-1].answer
    return AnswerAssessment(
        star=StarFlags(
            situation="situation" in answer,
            task="task" in answer,
            action="action" in answer,
            result="result" in answer,
            quantified_result="quantified" in answer,
        ),
        sufficient_signal="enough" in answer,
        claims_worth_chasing=[],
    )


def acme_context(target_level: TargetLevel | None = "senior") -> RoleContext:
    return RoleContext(
        company="Acme",
        role_title="Senior Backend Engineer",
        competencies=[
            CompetencyWeight(name="ownership", weight=5),
            CompetencyWeight(name="conflict", weight=4),
        ],
        target_level=target_level,
    )


def builder_returning(context: RoleContext) -> RoleBuilder:
    def build(posting_text: str) -> RoleContext:
        return context

    return build


class ScriptedDirector:
    """Replays scripted decisions; repeats the last one when the script runs out."""

    def __init__(self, decisions: Sequence[DirectorDecision]) -> None:
        self.decisions = list(decisions)
        self.situations: list[DirectorSituation] = []

    def decide(self, situation: DirectorSituation) -> DirectorDecision:
        self.situations.append(situation)
        index = min(len(self.situations) - 1, len(self.decisions) - 1)
        return self.decisions[index]


class StubInterviewer:
    def __init__(self) -> None:
        self.seen_briefs: list[str] = []
        self.seen_decisions: list[DirectorDecision] = []

    def question_for(
        self,
        decision: DirectorDecision,
        profile: CandidateProfile,
        context: RoleContext,
        brief: str,
        transcript: Sequence[Turn],
    ) -> str:
        self.seen_decisions.append(decision)
        self.seen_briefs.append(brief)
        return f"Question {len(self.seen_decisions)} about {decision.subject}."

    def closing_turn(self, transcript: Sequence[Turn]) -> str:
        return CLOSING_TURN


def stub_researcher(posting_text: str, context: RoleContext) -> str:
    return BRIEF


GRADE = SessionGrade(
    scores=[
        AnswerScore(
            question="Question 1 about their background.",
            star=StarFlags(
                situation=True, task=True, action=True, result=True, quantified_result=True
            ),
            specificity="high",
            ownership="clear",
            weak_or_missing=[],
            gap="",
            rationale="Complete STAR with a measured outcome at the target level.",
            score=4,
        )
    ]
)


class RecordingGrader:
    def __init__(self, grade: SessionGrade = GRADE) -> None:
        self.grade = grade
        self.seen: list[tuple[tuple[Turn, ...], TargetLevel]] = []

    def __call__(self, transcript: Sequence[Turn], target_level: TargetLevel) -> SessionGrade:
        self.seen.append((tuple(transcript), target_level))
        return self.grade


COACH_REPORT = CoachReport(
    summary="Solid stories, but you keep stopping before the outcome.",
    answer_advice=[
        AnswerAdvice(
            question="Question 1 about their background.",
            diagnosis="You named the work but not how it landed.",
            fix="End the story with the number you measured after.",
        )
    ],
    drills=[Drill(focus="Stating results", exercise="Retell a project ending on a metric.")],
    study_plan="Turn each project into a STAR story that ends on a number.",
)


class RecordingCoacher:
    def __init__(self, report: CoachReport = COACH_REPORT) -> None:
        self.report = report
        self.seen: list[tuple[tuple[Turn, ...], SessionGrade, TargetLevel]] = []

    def __call__(
        self, transcript: Sequence[Turn], grade: SessionGrade, target_level: TargetLevel
    ) -> CoachReport:
        self.seen.append((tuple(transcript), grade, target_level))
        return self.report


def build_engine(
    data_dir: Path,
    director: Director | None = None,
    interviewer: StubInterviewer | None = None,
    role_builder: RoleBuilder | None = None,
    researcher: object = None,
    grader: object = None,
    coacher: object = None,
    question_cap: int = 20,
    follow_up_cap: int = 6,
) -> InterviewEngine:
    return InterviewEngine(
        data_dir=data_dir,
        profile_parser=stub_parser,
        assessor=keyword_assessor,
        director=director or ScriptedDirector([OPENING_DECISION, WRAP_UP_DECISION]),
        interviewer=interviewer or StubInterviewer(),
        role_builder=role_builder or builder_returning(acme_context()),
        researcher=researcher or stub_researcher,  # type: ignore[arg-type]
        grader=grader or RecordingGrader(),  # type: ignore[arg-type]
        coacher=coacher or RecordingCoacher(),  # type: ignore[arg-type]
        question_cap=question_cap,
        follow_up_cap=follow_up_cap,
    )


def write_cv(tmp_path: Path) -> Path:
    cv = tmp_path / "cv.md"
    cv.write_text("# Jane Doe\nEngineer at Acme")
    return cv


def start_past_setup(
    engine: InterviewEngine, cv: Path, posting_text: str | None = None
) -> SessionSnapshot:
    session = engine.start(cv, posting_text=posting_text)
    if session.needs_level:
        session = engine.submit_level(session.thread_id, "mid")
    return session


def test_start_poses_the_directors_opening_topic(tmp_path: Path) -> None:
    with build_engine(tmp_path / "data") as engine:
        session = start_past_setup(engine, write_cv(tmp_path))

    assert session.question == "Question 1 about their background."
    assert session.thread_id
    assert session.profile == STUB_PROFILE


def test_without_a_posting_the_level_is_asked_never_defaulted(tmp_path: Path) -> None:
    with build_engine(tmp_path / "data") as engine:
        session = engine.start(write_cv(tmp_path))

        assert session.needs_level
        assert session.question is None

        session = engine.submit_level(session.thread_id, "mid")

    assert not session.needs_level
    assert session.question == "Question 1 about their background."


def test_a_deduced_level_skips_the_setup_question(tmp_path: Path) -> None:
    builder = builder_returning(acme_context(target_level="senior"))
    with build_engine(tmp_path / "data", role_builder=builder) as engine:
        session = engine.start(write_cv(tmp_path), posting_text=POSTING)

    assert not session.needs_level
    assert session.question == "Question 1 about their background."


def test_a_posting_without_a_clear_level_pauses_to_ask(tmp_path: Path) -> None:
    builder = builder_returning(acme_context(target_level=None))
    with build_engine(tmp_path / "data", role_builder=builder) as engine:
        session = engine.start(write_cv(tmp_path), posting_text=POSTING)

        assert session.needs_level

        session = engine.submit_level(session.thread_id, "senior")

    assert session.question == "Question 1 about their background."


def test_the_level_ask_survives_a_restart(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with build_engine(data_dir) as engine:
        engine.start(write_cv(tmp_path))

    with build_engine(data_dir) as engine:
        resumed = engine.resume_latest()
        assert resumed.needs_level
        session = engine.submit_level(resumed.thread_id, "junior")

    assert session.question == "Question 1 about their background."


def test_a_wrap_up_decision_ends_the_session_with_the_closing_turn(tmp_path: Path) -> None:
    director = ScriptedDirector([OPENING_DECISION, WRAP_UP_DECISION])
    with build_engine(tmp_path / "data", director=director) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, "A strong, complete story.")

    assert result.finished
    assert result.closing == CLOSING_TURN


def test_a_finished_session_carries_the_grade_over_the_real_transcript(tmp_path: Path) -> None:
    director = ScriptedDirector([OPENING_DECISION, WRAP_UP_DECISION])
    grader = RecordingGrader()
    with build_engine(tmp_path / "data", director=director, grader=grader) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, "A strong, complete story.")

    assert result.finished
    assert result.grade == GRADE
    seen_transcript, seen_level = grader.seen[0]
    assert [turn.answer for turn in seen_transcript] == ["A strong, complete story."]
    assert seen_level == "mid"


def test_a_finished_session_carries_the_coaching_over_the_real_transcript_and_grade(
    tmp_path: Path,
) -> None:
    director = ScriptedDirector([OPENING_DECISION, WRAP_UP_DECISION])
    coacher = RecordingCoacher()
    with build_engine(tmp_path / "data", director=director, coacher=coacher) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, "A strong, complete story.")

    assert result.finished
    assert result.coach == COACH_REPORT
    seen_transcript, seen_grade, seen_level = coacher.seen[0]
    assert [turn.answer for turn in seen_transcript] == ["A strong, complete story."]
    assert seen_grade == GRADE
    assert seen_level == "mid"


def test_a_finished_session_carries_the_full_transcript(tmp_path: Path) -> None:
    director = ScriptedDirector([OPENING_DECISION, FOLLOW_UP_DECISION, WRAP_UP_DECISION])
    with build_engine(tmp_path / "data", director=director) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        engine.submit_answer(session.thread_id, "We migrated the billing pipeline.")
        result = engine.submit_answer(session.thread_id, "It cut latency by 40 percent.")

    assert result.finished
    assert [turn.answer for turn in result.transcript] == [
        "We migrated the billing pipeline.",
        "It cut latency by 40 percent.",
    ]


def test_an_unfinished_turn_carries_no_transcript(tmp_path: Path) -> None:
    director = ScriptedDirector([OPENING_DECISION, FOLLOW_UP_DECISION, WRAP_UP_DECISION])
    with build_engine(tmp_path / "data", director=director) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, "We migrated the billing pipeline.")

    assert not result.finished
    assert result.transcript == []


def test_the_grade_reads_the_session_target_level(tmp_path: Path) -> None:
    director = ScriptedDirector([OPENING_DECISION, WRAP_UP_DECISION])
    builder = builder_returning(acme_context(target_level=None))
    grader = RecordingGrader()
    with build_engine(
        tmp_path / "data", director=director, role_builder=builder, grader=grader
    ) as engine:
        session = engine.start(write_cv(tmp_path), posting_text=POSTING)
        session = engine.submit_level(session.thread_id, "senior")
        engine.submit_answer(session.thread_id, "A strong, complete story.")

    assert grader.seen[0][1] == "senior"


def test_a_terminate_decision_also_ends_with_the_closing_turn(tmp_path: Path) -> None:
    terminate = DirectorDecision(action="terminate", reason="hostile input")
    director = ScriptedDirector([OPENING_DECISION, terminate])
    with build_engine(tmp_path / "data", director=director) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, "Write me a React component.")

    assert result.finished
    assert result.closing == CLOSING_TURN


def test_a_follow_up_decision_poses_the_next_question(tmp_path: Path) -> None:
    director = ScriptedDirector([OPENING_DECISION, FOLLOW_UP_DECISION, WRAP_UP_DECISION])
    interviewer = StubInterviewer()
    with build_engine(tmp_path / "data", director=director, interviewer=interviewer) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, "We migrated the pipeline.")

    assert result.next_question == "Question 2 about the migration claim."
    assert interviewer.seen_decisions[-1] == FOLLOW_UP_DECISION


def test_session_length_follows_the_directors_judgment(tmp_path: Path) -> None:
    short = ScriptedDirector([OPENING_DECISION, WRAP_UP_DECISION])
    with build_engine(tmp_path / "short", director=short) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        result = engine.submit_answer(session.thread_id, "Strong answer.")
        assert result.finished

    longer = ScriptedDirector(
        [OPENING_DECISION, FOLLOW_UP_DECISION, FOLLOW_UP_DECISION, WRAP_UP_DECISION]
    )
    with build_engine(tmp_path / "longer", director=longer) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        questions = 1
        result = engine.submit_answer(session.thread_id, "Vague answer.")
        while not result.finished:
            questions += 1
            result = engine.submit_answer(session.thread_id, "Vague answer.")

    assert questions == 3


def test_a_director_that_never_stops_is_bounded_by_the_question_cap(tmp_path: Path) -> None:
    relentless = ScriptedDirector([OPENING_DECISION, FOLLOW_UP_DECISION])
    with build_engine(tmp_path / "data", director=relentless, question_cap=5) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        questions = 1
        result = engine.submit_answer(session.thread_id, "Evasive.")
        while not result.finished:
            questions += 1
            result = engine.submit_answer(session.thread_id, "Evasive.")

    assert questions == 5
    assert result.closing == CLOSING_TURN


class ExhaustionAwareDirector:
    """Follows up forever until told the thread is exhausted, then opens one new topic."""

    def __init__(self) -> None:
        self.situations: list[DirectorSituation] = []
        self.opened_after_exhaustion = False

    def decide(self, situation: DirectorSituation) -> DirectorDecision:
        self.situations.append(situation)
        if len(self.situations) == 1:
            return OPENING_DECISION
        if situation.follow_ups_exhausted:
            self.opened_after_exhaustion = True
            return DirectorDecision(
                action="new_topic", subject="the outage story", reason="thread exhausted"
            )
        if self.opened_after_exhaustion:
            return WRAP_UP_DECISION
        return FOLLOW_UP_DECISION


def test_a_director_that_never_leaves_a_topic_is_forced_to_wrap(tmp_path: Path) -> None:
    relentless = ScriptedDirector([OPENING_DECISION, FOLLOW_UP_DECISION])
    with build_engine(tmp_path / "data", director=relentless, follow_up_cap=2) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        questions = 1
        result = engine.submit_answer(session.thread_id, "Evasive.")
        while not result.finished:
            questions += 1
            result = engine.submit_answer(session.thread_id, "Evasive.")

    assert questions == 3
    assert result.closing == CLOSING_TURN
    assert relentless.situations[-1].follow_ups_exhausted
    assert not relentless.situations[-2].follow_ups_exhausted


def test_an_exhausted_thread_lets_the_director_open_a_new_topic(tmp_path: Path) -> None:
    director = ExhaustionAwareDirector()
    with build_engine(tmp_path / "data", director=director, follow_up_cap=1) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        engine.submit_answer(session.thread_id, "Background answer.")
        result = engine.submit_answer(session.thread_id, "Follow-up answer.")

        assert result.next_question == "Question 3 about the outage story."

        engine.submit_answer(session.thread_id, "Outage answer.")
        state = engine._graph.get_state({"configurable": {"thread_id": session.thread_id}})

    topics = [entry.topic for entry in state.values["assessments"]]
    assert topics == ["their background", "their background", "the outage story"]


def test_the_director_sees_the_consecutive_follow_up_count(tmp_path: Path) -> None:
    director = ScriptedDirector(
        [OPENING_DECISION, FOLLOW_UP_DECISION, FOLLOW_UP_DECISION, WRAP_UP_DECISION]
    )
    with build_engine(tmp_path / "data", director=director) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        engine.submit_answer(session.thread_id, "First answer.")
        engine.submit_answer(session.thread_id, "Second answer.")
        engine.submit_answer(session.thread_id, "Third answer.")

    assert [s.consecutive_follow_ups for s in director.situations] == [0, 0, 1, 2]
    assert all(s.follow_up_cap == 6 for s in director.situations)


def test_the_director_sees_assessments_with_their_topics(tmp_path: Path) -> None:
    director = ScriptedDirector([OPENING_DECISION, FOLLOW_UP_DECISION, WRAP_UP_DECISION])
    with build_engine(tmp_path / "data", director=director) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        engine.submit_answer(session.thread_id, "situation task action, still vague")
        engine.submit_answer(session.thread_id, "enough now")

    final_situation = director.situations[-1]
    assert [entry.topic for entry in final_situation.assessments] == [
        "their background",
        "their background",
    ]
    assert [entry.assessment.sufficient_signal for entry in final_situation.assessments] == [
        False,
        True,
    ]
    assert final_situation.questions_asked == 2


def test_a_follow_up_keeps_the_current_topic_for_the_assessor(tmp_path: Path) -> None:
    new_topic = DirectorDecision(action="new_topic", subject="the outage story", reason="next")
    director = ScriptedDirector([OPENING_DECISION, new_topic, FOLLOW_UP_DECISION, WRAP_UP_DECISION])
    with build_engine(tmp_path / "data", director=director) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        engine.submit_answer(session.thread_id, "Background answer.")
        engine.submit_answer(session.thread_id, "Outage answer.")
        engine.submit_answer(session.thread_id, "Follow-up answer.")

        state = engine._graph.get_state({"configurable": {"thread_id": session.thread_id}})

    topics = [entry.topic for entry in state.values["assessments"]]
    assert topics == ["their background", "the outage story", "the outage story"]


def test_with_a_posting_the_company_brief_reaches_director_and_interviewer(
    tmp_path: Path,
) -> None:
    director = ScriptedDirector([OPENING_DECISION, WRAP_UP_DECISION])
    interviewer = StubInterviewer()
    with build_engine(tmp_path / "data", director=director, interviewer=interviewer) as engine:
        start_past_setup(engine, write_cv(tmp_path), posting_text=POSTING)

    assert director.situations[0].brief == BRIEF
    assert interviewer.seen_briefs == [BRIEF]


def test_without_a_posting_there_is_no_research_and_no_brief(tmp_path: Path) -> None:
    def exploding_researcher(posting_text: str, context: RoleContext) -> str:
        raise AssertionError("research must not run without a posting")

    director = ScriptedDirector([OPENING_DECISION, WRAP_UP_DECISION])
    engine = build_engine(tmp_path / "data", director=director, researcher=exploding_researcher)
    with engine:
        start_past_setup(engine, write_cv(tmp_path))

    assert director.situations[0].brief == ""


def test_a_failed_research_step_never_kills_the_session(tmp_path: Path) -> None:
    def failing_researcher(posting_text: str, context: RoleContext) -> str:
        raise RuntimeError("network down")

    director = ScriptedDirector([OPENING_DECISION, WRAP_UP_DECISION])
    engine = build_engine(tmp_path / "data", director=director, researcher=failing_researcher)
    with engine:
        session = start_past_setup(engine, write_cv(tmp_path), posting_text=POSTING)

    assert session.question == "Question 1 about their background."
    assert director.situations[0].brief == ""


def test_the_submitted_level_drives_the_directors_emphasis(tmp_path: Path) -> None:
    director = ScriptedDirector([OPENING_DECISION, WRAP_UP_DECISION])
    builder = builder_returning(acme_context(target_level=None))
    with build_engine(tmp_path / "data", director=director, role_builder=builder) as engine:
        session = engine.start(write_cv(tmp_path), posting_text=POSTING)
        engine.submit_level(session.thread_id, "senior")

    assert "strategic leadership" in director.situations[0].emphasis
    assert "problem solving" in director.situations[0].emphasis


def test_a_junior_level_gets_the_smaller_emphasis(tmp_path: Path) -> None:
    director = ScriptedDirector([OPENING_DECISION, WRAP_UP_DECISION])
    with build_engine(tmp_path / "data", director=director) as engine:
        session = engine.start(write_cv(tmp_path))
        engine.submit_level(session.thread_id, "junior")

    assert director.situations[0].emphasis == ("problem solving", "delivery", "learning")


def test_pause_mid_session_resumes_from_the_same_question(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    director = ScriptedDirector([OPENING_DECISION, FOLLOW_UP_DECISION, WRAP_UP_DECISION])
    with build_engine(data_dir, director=director) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        probe = engine.submit_answer(session.thread_id, "We migrated the pipeline.")

    resumed_director = ScriptedDirector([WRAP_UP_DECISION])
    with build_engine(data_dir, director=resumed_director) as engine:
        resumed = engine.resume_latest()
        assert resumed.thread_id == session.thread_id
        assert resumed.question == probe.next_question
        result = engine.submit_answer(resumed.thread_id, "The full story, enough now.")

    assert result.finished


def test_checkpoint_roundtrip_deserializes_only_registered_types(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    data_dir = tmp_path / "data"
    director = ScriptedDirector([OPENING_DECISION, FOLLOW_UP_DECISION, WRAP_UP_DECISION])
    with caplog.at_level(logging.WARNING):
        with build_engine(data_dir, director=director) as engine:
            session = start_past_setup(engine, write_cv(tmp_path), posting_text=POSTING)
            engine.submit_answer(session.thread_id, "situation task")

        resumed_director = ScriptedDirector([WRAP_UP_DECISION])
        with build_engine(data_dir, director=resumed_director) as engine:
            resumed = engine.resume_latest()
            engine.submit_answer(resumed.thread_id, "enough now")

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
        assessor=keyword_assessor,
        director=ScriptedDirector([WRAP_UP_DECISION]),
        interviewer=StubInterviewer(),
        role_builder=builder_returning(acme_context()),
        researcher=stub_researcher,
        grader=RecordingGrader(),
        coacher=RecordingCoacher(),
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
        engine.submit_answer(session.thread_id, "Strong answer.")

        with pytest.raises(EngineError, match="already finished"):
            engine.resume_latest()


def test_snapshot_latest_keeps_the_answered_turns_mid_session(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    director = ScriptedDirector([OPENING_DECISION, FOLLOW_UP_DECISION, WRAP_UP_DECISION])
    with build_engine(data_dir, director=director) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        probe = engine.submit_answer(session.thread_id, "We migrated the pipeline.")

    with build_engine(data_dir, director=ScriptedDirector([WRAP_UP_DECISION])) as engine:
        snapshot = engine.snapshot_latest()

    assert not snapshot.finished
    assert snapshot.question == probe.next_question
    assert snapshot.transcript == [
        Turn(question=session.question, answer="We migrated the pipeline.")
    ]


def test_snapshot_latest_recovers_a_finished_session(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with build_engine(data_dir) as engine:
        session = start_past_setup(engine, write_cv(tmp_path))
        engine.submit_answer(session.thread_id, "Strong answer.")

    with build_engine(data_dir) as engine:
        snapshot = engine.snapshot_latest()

    assert snapshot.finished
    assert snapshot.question is None
    assert snapshot.transcript
    assert snapshot.closing is not None
    assert snapshot.grade is not None
    assert snapshot.coach is not None

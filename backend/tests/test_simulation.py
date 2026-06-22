import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from sotellme.assessor import AnswerAssessment, StarFlags, TopicAssessment
from sotellme.catalog import ModelPrice
from sotellme.director import DirectorDecision, DirectorSituation
from sotellme.engine import SessionSnapshot, TurnResult
from sotellme.grader import AnswerScore, SessionGrade
from sotellme.interviewer import Turn
from sotellme.judge import (
    CompetencyCoverage,
    CoverageVerdict,
    QuestionContext,
    QuestionVerdict,
)
from sotellme.personas import Persona
from sotellme.pricing import ModelUsage
from sotellme.profile import CandidateProfile, Role
from sotellme.role import CompetencyWeight, RoleContext, TargetLevel
from sotellme.simulation import (
    DEFAULT_COST_GATE_THRESHOLD,
    QuestionRecord,
    RecordingDirector,
    RecordingInterviewer,
    SimulatedSession,
    TurnRecorder,
    confirm_run,
    estimate_run_cost,
    judge_session,
    replay_sessions,
    simulate_session,
    write_persona_cv,
    write_session_artifact,
)


def _persona(**overrides: object) -> Persona:
    base: dict[str, object] = {
        "name": "senior-strong",
        "target_level": "senior",
        "cv": "Naoki Brennan, senior engineer.",
        "posting": "Senior backend engineer.",
        "profile": "Complete STAR stories.",
        "base_behavior": "complete_star",
        "planted_turns": [],
    }
    base.update(overrides)
    return Persona.model_validate(base)


_PROFILE = CandidateProfile(
    roles=[Role(title="Senior Engineer", organization="Arcwell")],
    projects=[],
    quantified_claims=[],
    technologies=[],
)


class FakeEngine:
    """Mimics the InterviewEngine seam: needs a level, then poses questions until finished."""

    def __init__(self, questions: list[str]) -> None:
        self._questions = questions
        self._posed = 0
        self.answers: list[str] = []

    def start(self, cv_path: Path, posting_text: str | None = None) -> SessionSnapshot:
        return SessionSnapshot(thread_id="t", profile=_PROFILE, needs_level=True)

    def submit_level(self, thread_id: str, level: str) -> SessionSnapshot:
        question = self._questions[0]
        self._posed = 1
        return SessionSnapshot(
            thread_id=thread_id, profile=_PROFILE, needs_level=False, question=question
        )

    def submit_answer(self, thread_id: str, answer: str) -> TurnResult:
        self.answers.append(answer)
        transcript = [
            Turn(question=self._questions[i], answer=self.answers[i]) for i in range(self._posed)
        ]
        if self._posed >= len(self._questions):
            return TurnResult(next_question=None, closing="Thanks.", transcript=transcript)
        next_question = self._questions[self._posed]
        self._posed += 1
        return TurnResult(next_question=next_question, transcript=transcript)


class EchoSimulator:
    def answer(self, persona: Persona, question: str, transcript: Sequence[Turn]) -> str:
        return f"answer to: {question}"


def test_simulate_session_drives_the_interview_to_completion(tmp_path: Path) -> None:
    engine = FakeEngine(["Q1", "Q2", "Q3"])

    session = simulate_session(
        engine, EchoSimulator(), _persona(), tmp_path / "cv.md", TurnRecorder(), max_turns=20
    )

    assert engine.answers == ["answer to: Q1", "answer to: Q2", "answer to: Q3"]
    assert [t.question for t in session.transcript] == ["Q1", "Q2", "Q3"]
    assert session.finished_reason == "completed"
    assert session.persona == "senior-strong"


def test_simulate_session_records_the_thread_id_for_replay(tmp_path: Path) -> None:
    engine = FakeEngine(["Q1", "Q2"])

    session = simulate_session(
        engine, EchoSimulator(), _persona(), tmp_path / "cv.md", TurnRecorder(), max_turns=20
    )

    assert session.thread_id == "t"


class SpySimulator:
    def __init__(self) -> None:
        self.seen_transcript_lengths: list[int] = []

    def answer(self, persona: Persona, question: str, transcript: Sequence[Turn]) -> str:
        self.seen_transcript_lengths.append(len(transcript))
        return "answer"


def test_simulate_session_feeds_a_growing_transcript_to_the_simulator(tmp_path: Path) -> None:
    engine = FakeEngine(["Q1", "Q2", "Q3"])
    spy = SpySimulator()

    simulate_session(engine, spy, _persona(), tmp_path / "cv.md", TurnRecorder(), max_turns=20)

    assert spy.seen_transcript_lengths == [0, 1, 2]


def test_simulate_session_stops_at_the_max_turn_cap(tmp_path: Path) -> None:
    engine = FakeEngine(["Q1", "Q2", "Q3", "Q4", "Q5"])

    session = simulate_session(
        engine, EchoSimulator(), _persona(), tmp_path / "cv.md", TurnRecorder(), max_turns=2
    )

    assert session.turns == 2
    assert session.finished_reason == "max_turns"
    assert len(engine.answers) == 2


_CONTEXT = RoleContext(
    competencies=[CompetencyWeight(name="ownership", weight=5)], target_level="senior"
)


def _situation(**overrides: object) -> DirectorSituation:
    base: dict[str, object] = dict(
        profile=_PROFILE,
        context=_CONTEXT,
        emphasis=("ownership",),
        brief="",
        transcript=[Turn(question="Tell me about a project.", answer="We cut latency 38%.")],
        assessments=[
            TopicAssessment(
                topic="route optimization",
                assessment=AnswerAssessment(
                    star=StarFlags(
                        situation=True,
                        task=True,
                        action=False,
                        result=True,
                        quantified_result=True,
                    ),
                    sufficient_signal=True,
                    claims_worth_chasing=["cut latency 38%"],
                ),
            )
        ],
        questions_asked=1,
        question_cap=20,
        consecutive_follow_ups=2,
    )
    base.update(overrides)
    return DirectorSituation(**base)  # type: ignore[arg-type]


class StubDirector:
    def __init__(self, decision: DirectorDecision) -> None:
        self._decision = decision

    def decide(self, situation: DirectorSituation) -> DirectorDecision:
        return self._decision


class StubInterviewer:
    def question_for(self, decision, profile, context, brief, transcript) -> str:  # type: ignore[no-untyped-def]
        return "What did YOU personally change?"

    def closing_turn(self, transcript) -> str:  # type: ignore[no-untyped-def]
        return "Thanks."

    def redirect_turn(self, question) -> str:  # type: ignore[no-untyped-def]
        return "Let's get back to it."


def test_the_recorders_capture_a_question_with_its_director_context() -> None:
    recorder = TurnRecorder()
    decision = DirectorDecision(
        action="follow_up", subject="cut latency 38%", reason="ownership is blurred"
    )
    director = RecordingDirector(StubDirector(decision), recorder)
    interviewer = RecordingInterviewer(StubInterviewer(), recorder)
    situation = _situation()

    out = director.decide(situation)
    question = interviewer.question_for(out, _PROFILE, _CONTEXT, "", situation.transcript)

    assert question == "What did YOU personally change?"
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record.question == "What did YOU personally change?"
    assert record.competency == "cut latency 38%"
    assert "ownership is blurred" in record.gap
    assert record.target_level == "senior"
    assert record.sufficient_signal is True
    assert record.consecutive_follow_ups == 2


def test_closing_and_redirect_turns_are_not_recorded_as_questions() -> None:
    recorder = TurnRecorder()
    interviewer = RecordingInterviewer(StubInterviewer(), recorder)

    interviewer.closing_turn([])
    interviewer.redirect_turn("Q1")

    assert recorder.records == []


# Anthropic-shaped prices: smart (Opus) input $5 / output $25, fast (Sonnet) input $3 / output $15.
_PRICES = {
    "fast": ModelPrice(input=3.0, output=15.0),
    "smart": ModelPrice(input=5.0, output=25.0),
}


def test_estimate_run_cost_splits_fast_and_smart_and_scales_with_personas() -> None:
    one = estimate_run_cost(1, 10, "fast", "smart", _PRICES)
    five = estimate_run_cost(5, 10, "fast", "smart", _PRICES)

    assert one.usd is not None and one.fast_usd is not None and one.smart_usd is not None
    assert one.fast_usd > 0 and one.smart_usd > 0
    assert one.usd == pytest.approx(one.fast_usd + one.smart_usd)
    assert five.usd == pytest.approx(one.usd * 5)


def test_estimate_run_cost_prices_the_mix_well_below_an_all_smart_ceiling() -> None:
    estimate = estimate_run_cost(1, 8, "fast", "smart", _PRICES)

    assert estimate.usd is not None
    assert 0.8 < estimate.usd < 1.8


def test_estimate_run_cost_is_none_when_a_model_has_no_price() -> None:
    assert estimate_run_cost(3, 10, "fast", "missing", _PRICES).usd is None


def test_confirm_run_proceeds_without_asking_under_the_threshold() -> None:
    estimate = estimate_run_cost(1, 1, "fast", "smart", _PRICES)
    asked: list[str] = []

    def record(prompt: str) -> str:
        asked.append(prompt)
        return "n"

    proceeded = confirm_run(estimate, threshold=1000.0, input_fn=record)

    assert proceeded is True
    assert asked == []


def test_confirm_run_asks_above_the_threshold_and_respects_a_no() -> None:
    estimate = estimate_run_cost(50, 15, "fast", "smart", _PRICES)

    assert estimate.usd is not None and estimate.usd > DEFAULT_COST_GATE_THRESHOLD
    assert confirm_run(estimate, input_fn=lambda _: "n") is False
    assert confirm_run(estimate, input_fn=lambda _: "y") is True


def test_confirm_run_assume_yes_skips_the_prompt() -> None:
    estimate = estimate_run_cost(50, 15, "fast", "smart", _PRICES)

    def refuse(_: str) -> str:
        raise AssertionError("should not prompt when assume_yes is set")

    assert confirm_run(estimate, assume_yes=True, input_fn=refuse) is True


def test_write_session_artifact_is_idempotent(tmp_path: Path) -> None:
    session = SimulatedSession(
        persona="senior-strong",
        target_level="senior",
        transcript=[Turn(question="Q1", answer="A1")],
    )

    first = write_session_artifact(session, tmp_path)
    second = write_session_artifact(session, tmp_path)

    assert first == second
    assert list(tmp_path.glob("*.json")) == [first]
    reloaded = SimulatedSession.model_validate_json(first.read_text())
    assert reloaded.persona == "senior-strong"


def test_write_persona_cv_writes_the_synthetic_cv_to_a_file(tmp_path: Path) -> None:
    persona = _persona(cv="Synthetic CV body.")

    path = write_persona_cv(persona, tmp_path)

    assert path.read_text() == "Synthetic CV body."
    assert path.parent == tmp_path


class StubJudge:
    def __init__(
        self, question_verdict: QuestionVerdict, coverage_verdict: CoverageVerdict
    ) -> None:
        self._q = question_verdict
        self._c = coverage_verdict
        self.judged_contexts: list[QuestionContext] = []

    def judge_question(self, context: QuestionContext) -> QuestionVerdict:
        self.judged_contexts.append(context)
        return self._q

    def judge_coverage(
        self, target_level: TargetLevel, transcript: Sequence[Turn]
    ) -> CoverageVerdict:
        return self._c


def _question_verdict(score: int) -> QuestionVerdict:
    return QuestionVerdict(
        relevance_rationale="r",
        relevance_score=score,
        probes_the_flagged_gap_rationale="r",
        probes_the_flagged_gap_score=score,
        level_appropriateness_rationale="r",
        level_appropriateness_score=score,
        non_leading_rationale="r",
        non_leading_score=score,
        follow_up_discipline_rationale="r",
        follow_up_discipline_score=score,
        overall_rationale="ok",
        overall="good",
    )


def test_judge_session_judges_every_question_and_coverage() -> None:
    session = SimulatedSession(
        persona="senior-strong",
        target_level="senior",
        transcript=[Turn(question="Q1", answer="A1"), Turn(question="Q2", answer="A2")],
        questions=[
            QuestionRecord(question="Q1", competency="ownership", target_level="senior", gap="g1"),
            QuestionRecord(question="Q2", competency="impact", target_level="senior", gap="g2"),
        ],
    )
    coverage = CoverageVerdict(
        competencies=[CompetencyCoverage(competency="ownership", status="covered")],
        rationale="ok",
        verdict="good",
    )
    judge = StubJudge(_question_verdict(4), coverage)

    judgement = judge_session(judge, session)

    assert len(judgement.questions) == 2
    assert len(judge.judged_contexts) == 2
    assert judgement.coverage == coverage
    assert judgement.dimension_means["relevance"] == 4.0
    assert judgement.dimension_means["follow_up_discipline"] == 4.0


def _grade(score: int) -> SessionGrade:
    return SessionGrade(
        scores=[
            AnswerScore(
                question="Q1",
                turn_index=1,
                star=StarFlags(
                    situation=True, task=True, action=True, result=True, quantified_result=True
                ),
                specificity="high",
                ownership="clear",
                weak_or_missing=[],
                gap="" if score == 5 else "One refinement short of a five.",
                rationale="r",
                score=score,
            )
        ]
    )


class FakeReplayEngine:
    def __init__(self, grade: SessionGrade) -> None:
        self._grade = grade
        self.replayed: list[tuple[str, str]] = []

    def replay_from(self, thread_id: str, node: str) -> TurnResult:
        self.replayed.append((thread_id, node))
        return TurnResult(next_question=None, grade=self._grade, coach=None, transcript=[])

    def session_usage(self) -> dict[str, ModelUsage]:
        return {}


def test_replay_sessions_regrades_stored_sessions_and_skips_the_ungradable(tmp_path: Path) -> None:
    artifacts = tmp_path / "sessions"
    write_session_artifact(
        SimulatedSession(
            persona="senior-strong", target_level="senior", thread_id="t1", grade=_grade(2)
        ),
        artifacts,
    )
    write_session_artifact(
        SimulatedSession(
            persona="junior-thin",
            target_level="junior",
            thread_id="t2",
            grade=None,
            finished_reason="max_turns",
        ),
        artifacts,
    )
    write_session_artifact(
        SimulatedSession(persona="mid-pre", target_level="mid", thread_id=None, grade=_grade(3)),
        artifacts,
    )
    personas = [
        _persona(name="senior-strong"),
        _persona(name="junior-thin", target_level="junior"),
        _persona(name="mid-pre", target_level="mid"),
        _persona(name="staff-missing", target_level="staff"),
    ]
    engine = FakeReplayEngine(_grade(4))

    report = replay_sessions(engine, personas, artifacts, _PRICES)

    assert engine.replayed == [("t1", "grade")]
    reloaded = SimulatedSession.model_validate_json((artifacts / "senior-strong.json").read_text())
    assert reloaded.grade is not None
    assert [s.score for s in reloaded.grade.scores] == [4]
    assert "junior-thin" in report and "never reached grading" in report
    assert "mid-pre" in report and "replay support" in report
    assert "staff-missing" in report and "no stored session" in report


def test_replay_tolerates_a_stored_grade_the_current_invariant_rejects(tmp_path: Path) -> None:
    artifacts = tmp_path / "sessions"
    write_session_artifact(
        SimulatedSession(
            persona="senior-strong", target_level="senior", thread_id="t1", grade=_grade(4)
        ),
        artifacts,
    )
    # Rewrite the stored grade into one an older grader produced: a sub-5 with an empty gap,
    # which the current AnswerScore invariant rejects on load. Replay must re-grade it, not crash.
    path = artifacts / "senior-strong.json"
    raw = json.loads(path.read_text())
    raw["grade"]["scores"][0]["gap"] = ""
    path.write_text(json.dumps(raw))
    engine = FakeReplayEngine(_grade(5))

    report = replay_sessions(engine, [_persona(name="senior-strong")], artifacts, _PRICES)

    assert engine.replayed == [("t1", "grade")]
    reloaded = SimulatedSession.model_validate_json(path.read_text())
    assert reloaded.grade is not None
    assert [s.score for s in reloaded.grade.scores] == [5]
    assert "replay senior-strong" in report


def test_replay_sessions_passes_the_stage_through_to_the_engine(tmp_path: Path) -> None:
    artifacts = tmp_path / "sessions"
    write_session_artifact(
        SimulatedSession(
            persona="senior-strong", target_level="senior", thread_id="t1", grade=_grade(3)
        ),
        artifacts,
    )
    engine = FakeReplayEngine(_grade(3))

    report = replay_sessions(
        engine, [_persona(name="senior-strong")], artifacts, _PRICES, stage="coach"
    )

    assert engine.replayed == [("t1", "coach")]
    assert "recoach senior-strong" in report

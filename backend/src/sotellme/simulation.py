from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler

    from sotellme.config import ModelConfig
    from sotellme.engine import InterviewEngine

from sotellme.catalog import ModelPrice, default_catalog
from sotellme.coach import CoachReport
from sotellme.director import DirectorDecision, DirectorSituation
from sotellme.engine import SessionSnapshot, TurnResult
from sotellme.grader import SessionGrade
from sotellme.interviewer import Turn
from sotellme.judge import CoverageVerdict, QuestionContext, QuestionVerdict
from sotellme.personas import Persona
from sotellme.pricing import ModelUsage, cost_usd, format_cost_summary, summarize_actual_cost
from sotellme.profile import CandidateProfile
from sotellme.role import RoleContext, TargetLevel

# Per-session eval token model, split by which slot runs each agent so the estimate prices the
# real fast/smart mix instead of charging everything at the smart rate. Fast slot: CV parse,
# role context, and research (setup), and per turn the interviewer, answer assessor, guardrail,
# and the candidate-simulator answering. Smart slot: per turn the director and the per-question
# judge, and once at the end the grader, coach, and coverage judge. Calibrated against a real
# ~5-turn senior run (2026-06-16: 144k tokens, ~$0.68 on sonnet-fast / opus-smart); the figures
# scale with the turn count and stay a mild upper bound (they assume you hit the turn cap and
# credit no prompt caching).
EVAL_SETUP_FAST_INPUT = 18_000
EVAL_SETUP_FAST_OUTPUT = 2_500
EVAL_PER_TURN_FAST_INPUT = 14_000
EVAL_PER_TURN_FAST_OUTPUT = 900
EVAL_PER_TURN_SMART_INPUT = 8_000
EVAL_PER_TURN_SMART_OUTPUT = 1_200
EVAL_FEEDBACK_SMART_INPUT = 12_000
EVAL_FEEDBACK_SMART_OUTPUT = 4_000

# A simulated eval run with this estimated price or higher asks for explicit confirmation.
DEFAULT_COST_GATE_THRESHOLD = 3.50


@dataclass(frozen=True)
class RunCostEstimate:
    persona_count: int
    expected_turns: int
    fast_model: str
    smart_model: str
    fast_usd: float | None
    smart_usd: float | None
    usd: float | None


def eval_session_tokens(expected_turns: int) -> tuple[int, int, int, int]:
    """Returns (fast_input, fast_output, smart_input, smart_output) for one simulated session."""
    fast_input = EVAL_SETUP_FAST_INPUT + expected_turns * EVAL_PER_TURN_FAST_INPUT
    fast_output = EVAL_SETUP_FAST_OUTPUT + expected_turns * EVAL_PER_TURN_FAST_OUTPUT
    smart_input = expected_turns * EVAL_PER_TURN_SMART_INPUT + EVAL_FEEDBACK_SMART_INPUT
    smart_output = expected_turns * EVAL_PER_TURN_SMART_OUTPUT + EVAL_FEEDBACK_SMART_OUTPUT
    return fast_input, fast_output, smart_input, smart_output


def estimate_run_cost(
    persona_count: int,
    expected_turns: int,
    fast_model: str,
    smart_model: str,
    prices: Mapping[str, ModelPrice] | None = None,
) -> RunCostEstimate:
    if prices is None:
        prices = default_catalog().prices
    fast_price = prices.get(fast_model)
    smart_price = prices.get(smart_model)
    none = RunCostEstimate(persona_count, expected_turns, fast_model, smart_model, None, None, None)
    if fast_price is None or smart_price is None:
        return none
    fast_in, fast_out, smart_in, smart_out = eval_session_tokens(expected_turns)
    fast_usd = cost_usd(fast_price, fast_in, fast_out) * persona_count
    smart_usd = cost_usd(smart_price, smart_in, smart_out) * persona_count
    return RunCostEstimate(
        persona_count=persona_count,
        expected_turns=expected_turns,
        fast_model=fast_model,
        smart_model=smart_model,
        fast_usd=fast_usd,
        smart_usd=smart_usd,
        usd=fast_usd + smart_usd,
    )


def format_run_cost(estimate: RunCostEstimate) -> str:
    if estimate.usd is None:
        return (
            f"No price configured for {estimate.fast_model} / {estimate.smart_model}, "
            "so this eval run's cost can't be estimated."
        )
    return (
        f"Estimated eval-run cost: about ${estimate.usd:.2f} for {estimate.persona_count} "
        f"persona(s) at up to ~{estimate.expected_turns} questions each: "
        f"${estimate.fast_usd:.2f} on {estimate.fast_model} + "
        f"${estimate.smart_usd:.2f} on {estimate.smart_model} "
        f"(rough upper bound; real runs wrap earlier and hit the prompt cache)."
    )


def confirm_run(
    estimate: RunCostEstimate,
    threshold: float = DEFAULT_COST_GATE_THRESHOLD,
    assume_yes: bool = False,
    input_fn: Callable[[str], str] = input,
) -> bool:
    if assume_yes:
        return True
    if estimate.usd is not None and estimate.usd <= threshold:
        return True
    figure = f"${estimate.usd:.2f}" if estimate.usd is not None else "an unknown amount"
    reply = input_fn(
        f"This eval run is estimated at {figure} (gate: ${threshold:.2f}). Proceed? [y/N] "
    )
    return reply.strip().lower() in ("y", "yes")


class SessionEngine(Protocol):
    def start(self, cv_path: Path, posting_text: str | None = None) -> SessionSnapshot: ...

    def submit_level(self, thread_id: str, level: TargetLevel) -> SessionSnapshot: ...

    def submit_answer(self, thread_id: str, answer: str) -> TurnResult: ...


class Simulator(Protocol):
    def answer(self, persona: Persona, question: str, transcript: Sequence[Turn]) -> str: ...


class ReplayEngine(Protocol):
    def replay_from(self, thread_id: str, node: str) -> TurnResult: ...

    def session_usage(self) -> dict[str, ModelUsage]: ...


class QuestionRecord(BaseModel):
    question: str
    competency: str
    target_level: TargetLevel
    gap: str
    transcript: list[Turn] = []
    sufficient_signal: bool = False
    consecutive_follow_ups: int = 0

    def to_context(self) -> QuestionContext:
        return QuestionContext(
            question=self.question,
            competency=self.competency,
            target_level=self.target_level,
            gap=self.gap,
            transcript=self.transcript,
            sufficient_signal=self.sufficient_signal,
            consecutive_follow_ups=self.consecutive_follow_ups,
        )


class SimulatedSession(BaseModel):
    persona: str
    target_level: TargetLevel
    thread_id: str | None = None
    transcript: list[Turn] = []
    questions: list[QuestionRecord] = []
    closing: str | None = None
    grade: SessionGrade | None = None
    coach: CoachReport | None = None
    turns: int = 0
    finished_reason: str = "completed"


class TurnRecorder:
    """Shared between the recording director and interviewer to capture each question's context."""

    def __init__(self) -> None:
        self.records: list[QuestionRecord] = []
        self._last_situation: DirectorSituation | None = None

    def note_situation(self, situation: DirectorSituation) -> None:
        self._last_situation = situation

    def note_question(
        self,
        question: str,
        decision: DirectorDecision,
        context: RoleContext,
        transcript: Sequence[Turn],
    ) -> None:
        situation = self._last_situation
        assessments = situation.assessments if situation else []
        sufficient = bool(assessments) and assessments[-1].assessment.sufficient_signal
        self.records.append(
            QuestionRecord(
                question=question,
                competency=decision.subject or "general",
                target_level=context.target_level or "mid",
                gap=f"{decision.subject}: {decision.reason}".strip(": "),
                transcript=list(transcript),
                sufficient_signal=sufficient,
                consecutive_follow_ups=situation.consecutive_follow_ups if situation else 0,
            )
        )


QUESTION_DIMENSIONS = (
    "relevance",
    "probes_the_flagged_gap",
    "level_appropriateness",
    "non_leading",
    "follow_up_discipline",
)


class Judge(Protocol):
    def judge_question(self, context: QuestionContext) -> QuestionVerdict: ...

    def judge_coverage(
        self, target_level: TargetLevel, transcript: Sequence[Turn]
    ) -> CoverageVerdict: ...


class JudgedQuestion(BaseModel):
    question: str
    competency: str
    verdict: QuestionVerdict


class SessionJudgement(BaseModel):
    persona: str
    target_level: TargetLevel
    questions: list[JudgedQuestion] = []
    coverage: CoverageVerdict
    dimension_means: dict[str, float] = {}


def judge_session(judge: Judge, session: SimulatedSession) -> SessionJudgement:
    judged = [
        JudgedQuestion(
            question=record.question,
            competency=record.competency,
            verdict=judge.judge_question(record.to_context()),
        )
        for record in session.questions
    ]
    coverage = judge.judge_coverage(session.target_level, session.transcript)
    means: dict[str, float] = {}
    if judged:
        for dimension in QUESTION_DIMENSIONS:
            scores = [getattr(j.verdict, dimension).score for j in judged]
            means[dimension] = sum(scores) / len(scores)
    return SessionJudgement(
        persona=session.persona,
        target_level=session.target_level,
        questions=judged,
        coverage=coverage,
        dimension_means=means,
    )


def write_persona_cv(persona: Persona, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{persona.name}.md"
    path.write_text(persona.cv)
    return path


def build_recording_engine(
    config: "ModelConfig",
    callbacks: list["BaseCallbackHandler"],
    data_dir: Path,
    recorder: TurnRecorder,
) -> "InterviewEngine":
    """Build the real interview engine with the director and interviewer wrapped so each posed
    question is recorded with its context. The loop logic is the production loop, untouched."""
    from sotellme.cli import build_engine
    from sotellme.config import build_chat_model
    from sotellme.director import LLMDirector
    from sotellme.interviewer import LLMInterviewer

    director = RecordingDirector(
        LLMDirector(build_chat_model(config, "director"), config.agents["director"].provider),
        recorder,
    )
    interviewer = RecordingInterviewer(
        LLMInterviewer(
            build_chat_model(config, "interviewer"), config.agents["interviewer"].provider
        ),
        recorder,
    )
    return build_engine(
        config, callbacks, data_dir=data_dir, director=director, interviewer=interviewer
    )


def run_persona_simulation(
    persona: Persona,
    simulator: Simulator,
    config: "ModelConfig",
    callbacks: list["BaseCallbackHandler"],
    data_dir: Path,
    cv_dir: Path,
    max_turns: int,
) -> SimulatedSession:
    recorder = TurnRecorder()
    engine = build_recording_engine(config, callbacks, data_dir, recorder)
    cv_path = write_persona_cv(persona, cv_dir)
    try:
        return simulate_session(engine, simulator, persona, cv_path, recorder, max_turns)
    finally:
        engine.close()


def write_session_artifact(session: SimulatedSession, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{session.persona}.json"
    path.write_text(session.model_dump_json(indent=2))
    return path


class _Director(Protocol):
    def decide(self, situation: DirectorSituation) -> DirectorDecision: ...


class _Interviewer(Protocol):
    def question_for(
        self,
        decision: DirectorDecision,
        profile: CandidateProfile,
        context: RoleContext,
        brief: str,
        transcript: Sequence[Turn],
    ) -> str: ...

    def closing_turn(self, transcript: Sequence[Turn]) -> str: ...

    def redirect_turn(self, question: str) -> str: ...


class RecordingDirector:
    """Wraps the real director, stashing each situation for the recorder. Decisions pass through."""

    def __init__(self, inner: _Director, recorder: TurnRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def decide(self, situation: DirectorSituation) -> DirectorDecision:
        self._recorder.note_situation(situation)
        return self._inner.decide(situation)


class RecordingInterviewer:
    """Wraps the real interviewer, recording each posed question with its context. Output is
    the inner interviewer's, unchanged; closing and redirect turns are not recorded."""

    def __init__(self, inner: _Interviewer, recorder: TurnRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def question_for(
        self,
        decision: DirectorDecision,
        profile: CandidateProfile,
        context: RoleContext,
        brief: str,
        transcript: Sequence[Turn],
    ) -> str:
        question = self._inner.question_for(decision, profile, context, brief, transcript)
        self._recorder.note_question(question, decision, context, transcript)
        return question

    def closing_turn(self, transcript: Sequence[Turn]) -> str:
        return self._inner.closing_turn(transcript)

    def redirect_turn(self, question: str) -> str:
        return self._inner.redirect_turn(question)


def simulate_session(
    engine: SessionEngine,
    simulator: Simulator,
    persona: Persona,
    cv_path: Path,
    recorder: TurnRecorder,
    max_turns: int,
) -> SimulatedSession:
    snapshot = engine.start(cv_path, persona.posting)
    if snapshot.needs_level:
        snapshot = engine.submit_level(snapshot.thread_id, persona.target_level)
    thread_id = snapshot.thread_id

    transcript: list[Turn] = list(snapshot.transcript)
    question = snapshot.question
    turns = 0
    result: TurnResult | None = None
    while question is not None and turns < max_turns:
        answer = simulator.answer(persona, question, transcript)
        result = engine.submit_answer(thread_id, answer)
        turns += 1
        transcript = list(result.transcript)
        question = result.next_question

    finished_reason = "completed" if question is None else "max_turns"
    return SimulatedSession(
        persona=persona.name,
        target_level=persona.target_level,
        thread_id=thread_id,
        transcript=transcript,
        questions=recorder.records,
        closing=result.closing if result else None,
        grade=result.grade if result else None,
        coach=result.coach if result else None,
        turns=turns,
        finished_reason=finished_reason,
    )


def replay_skip_reason(session: SimulatedSession) -> str | None:
    if session.thread_id is None:
        return "stored before replay support; re-run it to capture its checkpoint"
    if session.grade is None:
        return f"never reached grading ({session.finished_reason}); no checkpoint to fork"
    return None


def _mean_score(grade: SessionGrade | None) -> str:
    if grade is None or not grade.scores:
        return "n/a"
    return f"{sum(score.score for score in grade.scores) / len(grade.scores):.2f}"


def format_replay_delta(
    persona: str, stage: str, before: SessionGrade | None, after: SessionGrade | None
) -> str:
    count = len(after.scores) if after else 0
    if stage == "coach":
        return (
            f"recoach {persona}: coach refreshed, grade kept at "
            f"{_mean_score(after)} ({count} answers)"
        )
    return (
        f"replay {persona}: mean score {_mean_score(before)} -> "
        f"{_mean_score(after)} ({count} answers)"
    )


def replay_sessions(
    engine: ReplayEngine,
    personas: Sequence[Persona],
    artifacts_dir: Path,
    prices: Mapping[str, ModelPrice],
    stage: str = "grade",
) -> str:
    lines: list[str] = []
    for persona in personas:
        path = artifacts_dir / f"{persona.name}.json"
        if not path.exists():
            lines.append(f"skip {persona.name}: no stored session; run it first")
            continue
        session = SimulatedSession.model_validate_json(path.read_text())
        reason = replay_skip_reason(session)
        if reason is not None:
            lines.append(f"skip {persona.name}: {reason}")
            continue
        assert session.thread_id is not None
        before = session.grade
        result = engine.replay_from(session.thread_id, stage)
        session.grade = result.grade
        session.coach = result.coach
        if result.closing:
            session.closing = result.closing
        write_session_artifact(session, artifacts_dir)
        lines.append(format_replay_delta(persona.name, stage, before, result.grade))
    cost = format_cost_summary(summarize_actual_cost(engine.session_usage(), prices))
    body = "\n".join(lines) if lines else "No sessions to replay."
    return f"{body}\n\n{cost}"

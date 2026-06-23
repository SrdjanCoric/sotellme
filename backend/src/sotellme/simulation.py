"""Drive, record, judge, and replay synthetic persona interview simulations."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ValidationError

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
    """Estimated cost of one simulated eval run, split by fast and smart model slots."""

    persona_count: int
    expected_turns: int
    fast_model: str
    smart_model: str
    fast_usd: float | None
    smart_usd: float | None
    usd: float | None


def eval_session_tokens(expected_turns: int) -> tuple[int, int, int, int]:
    """Compute the per-slot token estimate for one simulated session."""
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
    """Estimate the cost of a simulated eval run across the fast and smart slots.

    Args:
        persona_count: Number of personas the run covers.
        expected_turns: Expected number of turns per session.
        fast_model: Name of the model in the fast slot.
        smart_model: Name of the model in the smart slot.
        prices: Model price lookup; defaults to the default catalog's prices when None.

    Returns:
        A RunCostEstimate; its cost fields are None when either model's price is missing.
    """
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
    """Format a run cost estimate as a human-readable one-line summary."""
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
    """Confirm whether to proceed with a run, gating on its estimated cost."""
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
    """Engine surface needed to drive a single interview session forward."""

    def start(self, cv_path: Path, posting_text: str | None = None) -> SessionSnapshot:
        """Start a session from a CV and optional posting."""
        ...

    def submit_level(self, thread_id: str, level: TargetLevel) -> SessionSnapshot:
        """Submit the target level for a session awaiting one."""
        ...

    def submit_answer(self, thread_id: str, answer: str) -> TurnResult:
        """Submit a candidate answer and advance the session by one turn."""
        ...


class Simulator(Protocol):
    """Surface that produces a candidate's answer for a given question and transcript."""

    def answer(self, persona: Persona, question: str, transcript: Sequence[Turn]) -> str:
        """Produce the persona's answer to a question."""
        ...


class ReplayEngine(Protocol):
    """Engine surface needed to replay a stored session from a checkpoint."""

    def replay_from(self, thread_id: str, node: str) -> TurnResult:
        """Replay a session from a given node, re-running from that point on."""
        ...

    def session_usage(self) -> dict[str, ModelUsage]:
        """Return the model usage accumulated during the replay."""
        ...


class QuestionRecord(BaseModel):
    """A posed question captured with the director's context at the time it was asked."""

    question: str
    competency: str
    target_level: TargetLevel
    gap: str
    transcript: list[Turn] = []
    sufficient_signal: bool = False
    consecutive_follow_ups: int = 0

    def to_context(self) -> QuestionContext:
        """Convert this record into the context the judge consumes."""
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
    """A full simulated interview and its outcomes, serialized as an artifact."""

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
        """Stash the latest director situation for the next recorded question."""
        self._last_situation = situation

    def note_question(
        self,
        question: str,
        decision: DirectorDecision,
        context: RoleContext,
        transcript: Sequence[Turn],
    ) -> None:
        """Record a posed question, pairing it with the last stashed situation's context."""
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
    """Surface that scores individual questions and a session's competency coverage."""

    def judge_question(self, context: QuestionContext) -> QuestionVerdict:
        """Judge a single question's quality."""
        ...

    def judge_coverage(
        self, target_level: TargetLevel, transcript: Sequence[Turn]
    ) -> CoverageVerdict:
        """Judge how well a session covered the level's competencies."""
        ...


class JudgedQuestion(BaseModel):
    """A recorded question paired with the judge's verdict on it."""

    question: str
    competency: str
    verdict: QuestionVerdict


class SessionJudgement(BaseModel):
    """The judge's verdicts for a whole simulated session."""

    persona: str
    target_level: TargetLevel
    questions: list[JudgedQuestion] = []
    coverage: CoverageVerdict
    dimension_means: dict[str, float] = {}


TERMINATED_AS_EXPECTED_RATIONALE = (
    "Persona expected to be terminated by the guardrail and was; coverage not judged on the "
    "intentionally truncated transcript."
)


def judge_session(
    judge: Judge, session: SimulatedSession, expected_to_terminate: bool = False
) -> SessionJudgement:
    """Judge every question in a simulated session and its overall coverage.

    When the persona is expected to be terminated by the guardrail and the session ended that
    way, the truncated transcript is the intended outcome: score it a pass and skip the judge
    rather than penalizing the (correctly) thin coverage.
    """
    if expected_to_terminate and session.finished_reason == "terminated":
        return SessionJudgement(
            persona=session.persona,
            target_level=session.target_level,
            questions=[],
            coverage=CoverageVerdict(
                competencies=[], rationale=TERMINATED_AS_EXPECTED_RATIONALE, verdict="good"
            ),
            dimension_means={},
        )
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
            scores = [j.verdict.dimensions[dimension].score for j in judged]
            means[dimension] = sum(scores) / len(scores)
    return SessionJudgement(
        persona=session.persona,
        target_level=session.target_level,
        questions=judged,
        coverage=coverage,
        dimension_means=means,
    )


def write_persona_cv(persona: Persona, out_dir: Path) -> Path:
    """Write a persona's CV to a Markdown file, creating the directory if needed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{persona.name}.md"
    path.write_text(persona.cv)
    return path


def no_web_research(url: str) -> str:
    """Refuse every research fetch so simulations never hit the live web.

    Synthetic personas invent company names that can collide with real domains; a live
    fetch would ground the brief in the wrong same-named company. Refusing keeps the brief
    on the posting's own topics, which is the grounding the simulation is meant to test.
    """
    from sotellme.fetch import ResearchFetchError

    raise ResearchFetchError("Web research is disabled in simulations; use the posting alone.")


def build_recording_engine(
    config: "ModelConfig",
    callbacks: list["BaseCallbackHandler"],
    data_dir: Path,
    recorder: TurnRecorder,
) -> "InterviewEngine":
    """Build the real interview engine with the director and interviewer wrapped so each
    posed question is recorded with its context. The loop logic is the production loop,
    untouched."""
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
        config,
        callbacks,
        data_dir=data_dir,
        director=director,
        interviewer=interviewer,
        fetcher=no_web_research,
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
    """Run one full simulated interview for a persona, recording each question.

    Builds a recording engine, writes the persona's CV, runs the session, and closes the
    engine afterward.

    Args:
        persona: The persona to simulate.
        simulator: The simulator that produces the candidate's answers.
        config: The model configuration for the engine's agents.
        callbacks: Callback handlers attached to the engine.
        data_dir: Directory the engine persists session data into.
        cv_dir: Directory the persona's CV is written into.
        max_turns: Maximum number of turns to run.

    Returns:
        The simulated session with its transcript, questions, and outcomes.
    """
    recorder = TurnRecorder()
    engine = build_recording_engine(config, callbacks, data_dir, recorder)
    cv_path = write_persona_cv(persona, cv_dir)
    try:
        return simulate_session(engine, simulator, persona, cv_path, recorder, max_turns)
    finally:
        engine.close()


def write_session_artifact(session: SimulatedSession, out_dir: Path) -> Path:
    """Serialize a simulated session to a JSON artifact, creating the directory if needed."""
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
        """Stash the situation, then delegate the decision to the inner director."""
        self._recorder.note_situation(situation)
        return self._inner.decide(situation)


class RecordingInterviewer:
    """Wrap the real interviewer, recording each posed question with its context.

    Output is the inner interviewer's, unchanged; closing and redirect turns are not
    recorded.
    """

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
        """Produce a question via the inner interviewer and record it with its context."""
        question = self._inner.question_for(decision, profile, context, brief, transcript)
        self._recorder.note_question(question, decision, context, transcript)
        return question

    def closing_turn(self, transcript: Sequence[Turn]) -> str:
        """Delegate the closing turn to the inner interviewer without recording it."""
        return self._inner.closing_turn(transcript)

    def redirect_turn(self, question: str) -> str:
        """Delegate the redirect turn to the inner interviewer without recording it."""
        return self._inner.redirect_turn(question)


def simulate_session(
    engine: SessionEngine,
    simulator: Simulator,
    persona: Persona,
    cv_path: Path,
    recorder: TurnRecorder,
    max_turns: int,
) -> SimulatedSession:
    """Drive an engine through a persona's interview turn by turn until it ends.

    Starts the session, submits the level when requested, then answers each question via
    the simulator up to max_turns, recording the transcript and final outcomes.

    Args:
        engine: The session engine to drive.
        simulator: The simulator that produces the candidate's answers.
        persona: The persona being simulated.
        cv_path: Path to the persona's CV file.
        recorder: The recorder whose captured questions are attached to the session.
        max_turns: Maximum number of turns to run.

    Returns:
        The simulated session; finished_reason is "completed" when the engine ran out of
        questions, "max_turns" when the turn limit was reached first, or "terminated" when
        the guardrail ended the interview early.
    """
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

    if result is not None and result.ended_early:
        finished_reason = "terminated"
    else:
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


def replay_skip_reason(thread_id: str | None, was_graded: bool, finished_reason: str) -> str | None:
    """Explain why a stored session cannot be replayed, if it cannot."""
    if thread_id is None:
        return "stored before replay support; re-run it to capture its checkpoint"
    if not was_graded:
        return f"never reached grading ({finished_reason}); no checkpoint to fork"
    return None


def _load_replayable(path: Path) -> tuple[SimulatedSession, SessionGrade | None, bool]:
    """Load a stored session for replay, tolerating a grade an older grader produced.

    Replay overwrites the grade, so a stored grade the current AnswerScore invariant
    rejects must not block loading. The stale grade is dropped from the strict load and
    best-effort decoded only for the before/after delta; a flag records that the session
    was graded so the skip logic still forks it.
    """
    raw = json.loads(path.read_text())
    stored_grade = raw.pop("grade", None)
    session = SimulatedSession.model_validate(raw)
    before: SessionGrade | None = None
    if stored_grade is not None:
        try:
            before = SessionGrade.model_validate(stored_grade)
        except ValidationError:
            before = None
    return session, before, stored_grade is not None


def _mean_score(grade: SessionGrade | None) -> str:
    """Format a grade's mean answer score, or 'n/a' when there is none."""
    if grade is None or not grade.scores:
        return "n/a"
    return f"{sum(score.score for score in grade.scores) / len(grade.scores):.2f}"


def format_replay_delta(
    persona: str, stage: str, before: SessionGrade | None, after: SessionGrade | None
) -> str:
    """Format a one-line summary of how a replay changed a persona's grade."""
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
    """Replay each persona's stored session from a stage, rewriting its artifact.

    Skips personas with no stored session or one that cannot be replayed, replays the
    rest from the given stage, updates grade, coach, and closing on the artifact, and
    appends a cost summary.

    Args:
        engine: The replay engine used to fork and re-run sessions.
        personas: The personas whose stored sessions are replayed.
        artifacts_dir: Directory holding and receiving the session artifacts.
        prices: Model price lookup used to summarize replay cost.
        stage: The stage to replay from.

    Returns:
        A multi-line report of per-persona outcomes followed by a cost summary.
    """
    lines: list[str] = []
    for persona in personas:
        path = artifacts_dir / f"{persona.name}.json"
        if not path.exists():
            lines.append(f"skip {persona.name}: no stored session; run it first")
            continue
        session, before, was_graded = _load_replayable(path)
        reason = replay_skip_reason(session.thread_id, was_graded, session.finished_reason)
        if reason is not None:
            lines.append(f"skip {persona.name}: {reason}")
            continue
        assert session.thread_id is not None
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

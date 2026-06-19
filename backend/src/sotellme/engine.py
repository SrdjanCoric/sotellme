"""LangGraph interview engine that orchestrates the interview, grading, and coaching flow."""

import sqlite3
import uuid
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, TypedDict, get_args

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from sotellme.assessor import AnswerAssessment, StarFlags, TopicAssessment
from sotellme.budget import DEFAULT_TOKEN_BUDGET, BudgetCallback, default_session_budget
from sotellme.coach import AnswerAdvice, CoachReport, Drill
from sotellme.coverage import (
    DEFAULT_FOLLOW_UP_CAP,
    DEFAULT_QUESTION_CAP,
    EnvelopeState,
    follow_up_allowed,
    question_allowed,
)
from sotellme.director import DirectorDecision, DirectorSituation
from sotellme.extraction import extract_cv_text
from sotellme.grader import AnswerScore, SessionGrade
from sotellme.guardrail import GuardrailState, GuardrailVerdict, resolve_turn
from sotellme.interviewer import Turn
from sotellme.pricing import ModelUsage
from sotellme.profile import CandidateProfile, Project, Role
from sotellme.role import (
    CompetencyWeight,
    RoleContext,
    TargetLevel,
    default_role_context,
    level_emphasis,
)

CHECKPOINTED_TYPES = (
    CandidateProfile,
    Role,
    Project,
    Turn,
    RoleContext,
    CompetencyWeight,
    StarFlags,
    AnswerAssessment,
    TopicAssessment,
    DirectorDecision,
    AnswerScore,
    SessionGrade,
    AnswerAdvice,
    Drill,
    CoachReport,
)

ENVELOPE_WRAP_REASON = "The session envelope is closed; the interview ends here."

BUDGET_WRAP_REASON = "The session token budget is nearly spent; the interview wraps up here."

FOLLOW_UPS_EXHAUSTED_WRAP_REASON = (
    "Follow-ups on this topic are exhausted and the director offered no new topic; "
    "the interview ends here."
)

ProfileParser = Callable[[str], CandidateProfile]
Assessor = Callable[[str, Sequence[Turn]], AnswerAssessment]
RoleBuilder = Callable[[str], RoleContext]
Researcher = Callable[[str, RoleContext], str]
Grader = Callable[[Sequence[Turn], TargetLevel], SessionGrade]
Coacher = Callable[[Sequence[Turn], SessionGrade, TargetLevel], CoachReport]


class Director(Protocol):
    """Decides the next interview move from the current situation."""

    def decide(self, situation: DirectorSituation) -> DirectorDecision:
        """Decide the next interview action for the given situation"""
        ...


class Interviewer(Protocol):
    """Generates interviewer turns: questions, redirects, and the closing remark."""

    def question_for(
        self,
        decision: DirectorDecision,
        profile: CandidateProfile,
        context: RoleContext,
        brief: str,
        transcript: Sequence[Turn],
    ) -> str:
        """Produce the next interview question for the decision and session context"""
        ...

    def closing_turn(self, transcript: Sequence[Turn]) -> str:
        """Produce the interviewer's closing remark for the transcript"""
        ...

    def redirect_turn(self, question: str) -> str:
        """Produce a redirect prompt steering the candidate back to the question"""
        ...


class Guardrail(Protocol):
    """Classifies a question and answer into a guardrail verdict."""

    def classify(self, question: str, answer: str) -> GuardrailVerdict:
        """Classify a question and answer into a guardrail verdict"""
        ...


class EngineError(Exception):
    """Raised when an interview session cannot be started, resumed, or replayed."""

    pass


class InterviewState(TypedDict, total=False):
    """Mutable state threaded through the interview graph nodes."""

    cv_path: str
    cv_text: str
    posting_text: str
    profile: CandidateProfile
    role_context: RoleContext
    emphasis: list[str]
    company_brief: str
    question: str
    pending_answer: str
    redirect: str
    guardrail_redirects: int
    screened: GuardrailVerdict
    transcript: list[Turn]
    assessments: list[TopicAssessment]
    questions_asked: int
    consecutive_follow_ups: int
    current_topic: str
    decision: DirectorDecision
    closing: str
    grade: SessionGrade
    coach: CoachReport


class SessionSnapshot(BaseModel):
    """Point-in-time view of an interview session for callers to render or resume."""

    thread_id: str
    profile: CandidateProfile
    needs_level: bool
    level: TargetLevel | None = None
    question: str | None = None
    transcript: list[Turn] = []
    finished: bool = False
    closing: str | None = None
    grade: SessionGrade | None = None
    coach: CoachReport | None = None


class SessionListItem(BaseModel):
    """Summary row describing one stored session for listing."""

    thread_id: str
    company: str | None = None
    role_title: str | None = None
    target_level: TargetLevel | None = None
    finished: bool = False
    created_at: str | None = None


class TurnResult(BaseModel):
    """Outcome of submitting one answer to the engine."""

    next_question: str | None = None
    closing: str | None = None
    grade: SessionGrade | None = None
    coach: CoachReport | None = None
    transcript: list[Turn] = []

    @property
    def finished(self) -> bool:
        """Whether the session has finished, i.e. no next question remains"""
        return self.next_question is None


def _extract(state: InterviewState) -> InterviewState:
    """Extract the CV text from the CV path into state"""
    return {"cv_text": extract_cv_text(Path(state["cv_path"]))}


def _ask_level(state: InterviewState) -> InterviewState:
    """Interrupt to ask for the target level when the role context has none set"""
    context = state["role_context"]
    if context.target_level is not None:
        return {}
    level = interrupt({"ask_level": True})
    return {"role_context": context.model_copy(update={"target_level": level})}


def _derive_emphasis(state: InterviewState) -> InterviewState:
    """Derive the competency emphasis labels from the target level into state"""
    level = state["role_context"].target_level
    return {"emphasis": list(level_emphasis(level)) if level is not None else []}


def _await_answer(state: InterviewState) -> InterviewState:
    """Interrupt with the current prompt and store the resumed answer as pending"""
    prompt = state.get("redirect") or state["question"]
    answer = interrupt({"question": prompt})
    return {"pending_answer": str(answer)}


class InterviewEngine:
    """Stateful interview engine backed by a checkpointed LangGraph state machine."""

    def __init__(
        self,
        data_dir: Path,
        profile_parser: ProfileParser,
        assessor: Assessor,
        director: Director,
        interviewer: Interviewer,
        role_builder: RoleBuilder,
        researcher: Researcher,
        grader: Grader,
        coacher: Coacher,
        guardrail: Guardrail,
        question_cap: int = DEFAULT_QUESTION_CAP,
        follow_up_cap: int = DEFAULT_FOLLOW_UP_CAP,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir
        self._token_budget = token_budget
        self._budget_callback = BudgetCallback()
        self._callbacks = [*(callbacks or []), self._budget_callback]
        self._conn = sqlite3.connect(data_dir / "checkpoints.sqlite", check_same_thread=False)

        def parse_profile(state: InterviewState) -> InterviewState:
            return {"profile": profile_parser(state["cv_text"])}

        def build_context(state: InterviewState) -> InterviewState:
            posting = state.get("posting_text")
            context = role_builder(posting) if posting else default_role_context()
            return {"role_context": context}

        def research(state: InterviewState) -> InterviewState:
            posting = state.get("posting_text")
            if not posting:
                return {"company_brief": ""}
            try:
                brief = researcher(posting, state["role_context"])
            except Exception:  # research is an enhancement; it must never kill the session
                brief = ""
            return {"company_brief": brief}

        def direct(state: InterviewState) -> InterviewState:
            budget = default_session_budget(
                self._token_budget, tokens_used=self._budget_callback.total_tokens
            )
            envelope = EnvelopeState(
                questions_asked=state.get("questions_asked", 0),
                consecutive_follow_ups=state.get("consecutive_follow_ups", 0),
                budget_exhausted=budget.exhausted,
            )
            if not question_allowed(envelope, question_cap):
                reason = BUDGET_WRAP_REASON if budget.exhausted else ENVELOPE_WRAP_REASON
                return {"decision": DirectorDecision(action="wrap_up", reason=reason)}
            situation = DirectorSituation(
                profile=state["profile"],
                context=state["role_context"],
                emphasis=tuple(state.get("emphasis", [])),
                brief=state.get("company_brief", ""),
                transcript=state.get("transcript", []),
                assessments=state.get("assessments", []),
                questions_asked=state.get("questions_asked", 0),
                question_cap=question_cap,
                consecutive_follow_ups=envelope.consecutive_follow_ups,
                follow_up_cap=follow_up_cap,
            )
            decision = director.decide(situation)
            if decision.action == "follow_up" and not follow_up_allowed(envelope, follow_up_cap):
                decision = director.decide(replace(situation, follow_ups_exhausted=True))
                if decision.action == "follow_up":
                    decision = DirectorDecision(
                        action="wrap_up", reason=FOLLOW_UPS_EXHAUSTED_WRAP_REASON
                    )
            return {"decision": decision}

        def pose_question(state: InterviewState) -> InterviewState:
            decision = state["decision"]
            question = interviewer.question_for(
                decision,
                state["profile"],
                state["role_context"],
                state.get("company_brief", ""),
                state.get("transcript", []),
            )
            topic = (
                decision.subject
                if decision.action == "new_topic"
                else state.get("current_topic", "")
            )
            follow_ups = (
                state.get("consecutive_follow_ups", 0) + 1 if decision.action == "follow_up" else 0
            )
            return {
                "question": question,
                "questions_asked": state.get("questions_asked", 0) + 1,
                "consecutive_follow_ups": follow_ups,
                "current_topic": topic,
            }

        def screen(state: InterviewState) -> InterviewState:
            pending = state.get("pending_answer", "")
            raw = guardrail.classify(state["question"], pending)
            envelope = GuardrailState(consecutive_redirects=state.get("guardrail_redirects", 0))
            verdict, envelope = resolve_turn(raw, envelope)
            if verdict == "allow":
                turn = Turn(question=state["question"], answer=pending)
                return {
                    "transcript": [*state.get("transcript", []), turn],
                    "pending_answer": "",
                    "redirect": "",
                    "guardrail_redirects": 0,
                    "screened": "allow",
                }
            if verdict == "redirect":
                return {
                    "redirect": interviewer.redirect_turn(state["question"]),
                    "pending_answer": "",
                    "guardrail_redirects": envelope.consecutive_redirects,
                    "screened": "redirect",
                }
            return {"pending_answer": "", "redirect": "", "screened": "terminate"}

        def assess(state: InterviewState) -> InterviewState:
            topic = state.get("current_topic", "")
            assessment = assessor(topic, state["transcript"])
            record = TopicAssessment(topic=topic, assessment=assessment)
            return {"assessments": [*state.get("assessments", []), record]}

        def pose_closing(state: InterviewState) -> InterviewState:
            return {"closing": interviewer.closing_turn(state.get("transcript", []))}

        def grade(state: InterviewState) -> InterviewState:
            transcript = state.get("transcript", [])
            if not transcript:
                return {"grade": SessionGrade(scores=[])}
            level = state["role_context"].target_level or "mid"
            return {"grade": grader(transcript, level)}

        def coach(state: InterviewState) -> InterviewState:
            level = state["role_context"].target_level or "mid"
            return {"coach": coacher(state.get("transcript", []), state["grade"], level)}

        def route_after_direct(state: InterviewState) -> str:
            if state["decision"].action in ("wrap_up", "terminate"):
                return "pose_closing"
            return "pose_question"

        def route_after_screen(state: InterviewState) -> str:
            verdict = state["screened"]
            if verdict == "allow":
                return "assess"
            if verdict == "redirect":
                return "await_answer"
            return "pose_closing"

        graph = StateGraph(InterviewState)
        graph.add_node("extract", _extract)
        graph.add_node("parse_profile", parse_profile)
        graph.add_node("build_context", build_context)
        graph.add_node("ask_level", _ask_level)
        graph.add_node("derive_emphasis", _derive_emphasis)
        graph.add_node("research", research)
        graph.add_node("direct", direct)
        graph.add_node("pose_question", pose_question)
        graph.add_node("await_answer", _await_answer)
        graph.add_node("screen", screen)
        graph.add_node("assess", assess)
        graph.add_node("pose_closing", pose_closing)
        graph.add_node("grade", grade)
        graph.add_node("coach", coach)
        graph.add_edge(START, "extract")
        graph.add_edge("extract", "parse_profile")
        graph.add_edge("parse_profile", "build_context")
        graph.add_edge("build_context", "ask_level")
        graph.add_edge("ask_level", "derive_emphasis")
        graph.add_edge("derive_emphasis", "research")
        graph.add_edge("research", "direct")
        graph.add_conditional_edges("direct", route_after_direct, ["pose_question", "pose_closing"])
        graph.add_edge("pose_question", "await_answer")
        graph.add_edge("await_answer", "screen")
        graph.add_conditional_edges(
            "screen", route_after_screen, ["assess", "await_answer", "pose_closing"]
        )
        graph.add_edge("assess", "direct")
        graph.add_edge("pose_closing", "grade")
        graph.add_edge("grade", "coach")
        graph.add_edge("coach", END)
        serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINTED_TYPES)
        checkpointer = SqliteSaver(self._conn, serde=serde)
        checkpointer.setup()
        self._graph = graph.compile(checkpointer=checkpointer)

    def start(self, cv_path: Path, posting_text: str | None = None) -> SessionSnapshot:
        """Start a new interview session and run it up to the first interruption.

        A new thread is created and recorded as the latest session.

        Args:
            cv_path: Path to the candidate's CV.
            posting_text: Optional job posting text to build the role context and brief.

        Returns:
            A snapshot of the pending session.

        Raises:
            EngineError: If the started session is already finished or did not get far
                enough to reopen.
        """
        thread_id = uuid.uuid4().hex
        initial: InterviewState = {"cv_path": str(cv_path)}
        if posting_text is not None:
            initial["posting_text"] = posting_text
        self._graph.invoke(initial, self._config(thread_id))
        (self._data_dir / "latest").write_text(thread_id)
        return self._pending_session(thread_id)

    def resume_latest(self) -> SessionSnapshot:
        """Return a pending snapshot of the most recent session.

        Returns:
            A snapshot of the latest pending session.

        Raises:
            EngineError: If there is no session to resume, it is already finished, or it
                did not get far enough to reopen.
        """
        return self._pending_session(self._latest_thread())

    def snapshot_latest(self) -> SessionSnapshot:
        """Return a snapshot of the most recent session, finished or not.

        Returns:
            A snapshot of the latest session.

        Raises:
            EngineError: If there is no session, or it did not get far enough to reopen.
        """
        return self._snapshot(self._latest_thread())

    def snapshot(self, thread_id: str) -> SessionSnapshot:
        """Return a snapshot of a specific session.

        Args:
            thread_id: Identifier of the session's graph thread.

        Returns:
            A snapshot of the session.

        Raises:
            EngineError: If the session did not get far enough to reopen.
        """
        return self._snapshot(thread_id)

    def list_sessions(self, limit: int | None = None, offset: int = 0) -> list[SessionListItem]:
        """List stored sessions ordered from most to least recently checkpointed.

        Args:
            limit: Maximum number of sessions to return; None for no limit.
            offset: Number of sessions to skip from the start of the ordering.

        Returns:
            Summary items for the matching sessions.
        """
        rows = self._conn.execute(
            "SELECT thread_id FROM checkpoints "
            "GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC LIMIT ? OFFSET ?",
            (-1 if limit is None else limit, offset),
        ).fetchall()
        return [self._session_item(thread_id) for (thread_id,) in rows]

    def _session_item(self, thread_id: str) -> SessionListItem:
        """Build a summary list item from a session's checkpointed state"""
        state = self._graph.get_state(self._config(thread_id))
        role_context = state.values.get("role_context")
        return SessionListItem(
            thread_id=thread_id,
            company=role_context.company if role_context is not None else None,
            role_title=role_context.role_title if role_context is not None else None,
            target_level=role_context.target_level if role_context is not None else None,
            finished=not state.next,
            created_at=state.created_at,
        )

    def _latest_thread(self) -> str:
        """Read the latest session's thread id, raising if no session has been recorded"""
        latest = self._data_dir / "latest"
        if not latest.exists():
            raise EngineError("No session to resume. Start one with: sotellme interview")
        return latest.read_text().strip()

    def submit_level(self, thread_id: str, level: TargetLevel) -> SessionSnapshot:
        """Supply the target level to a session waiting for it and advance the graph.

        Args:
            thread_id: Identifier of the session's graph thread.
            level: The target level to apply.

        Returns:
            A snapshot of the pending session after advancing.

        Raises:
            EngineError: If the level is not a recognized target level, or the resulting
                session is already finished or did not get far enough to reopen.
        """
        if level not in get_args(TargetLevel):
            valid = ", ".join(get_args(TargetLevel))
            raise EngineError(f"Unknown level {level!r}: choose one of {valid}.")
        self._graph.invoke(Command(resume=level), self._config(thread_id))
        return self._pending_session(thread_id)

    def submit_answer(self, thread_id: str, answer: str) -> TurnResult:
        """Submit an answer to a session and advance to the next question or the end.

        Args:
            thread_id: Identifier of the session's graph thread.
            answer: The candidate's answer to the current question.

        Returns:
            The turn result: the next question (or redirect) when more remains, otherwise
            the closing remark, grade, and coaching report.
        """
        self._graph.invoke(Command(resume=answer), self._config(thread_id))
        state = self._graph.get_state(self._config(thread_id))
        if state.next:
            values = state.values
            return TurnResult(
                next_question=values.get("redirect") or values["question"],
                transcript=values.get("transcript", []),
            )
        return TurnResult(
            next_question=None,
            closing=state.values.get("closing"),
            grade=state.values.get("grade"),
            coach=state.values.get("coach"),
            transcript=state.values.get("transcript", []),
        )

    def replay_from(self, thread_id: str, node: str = "grade") -> TurnResult:
        """Re-run a finished session from the checkpoint just before a given node.

        Finds the most recent checkpoint whose next step includes the node, forks from it
        with the engine's callbacks, and re-runs to completion. Useful for re-grading or
        re-coaching a stored session.

        Args:
            thread_id: Identifier of the session's graph thread.
            node: The graph node to replay from; defaults to the grade node.

        Returns:
            The turn result after replaying to completion.

        Raises:
            EngineError: If the session has no checkpoint before the given node.
        """
        fork = next(
            (
                snapshot
                for snapshot in self._graph.get_state_history(self._config(thread_id))
                if node in snapshot.next
            ),
            None,
        )
        if fork is None:
            raise EngineError(f"Session {thread_id!r} has no checkpoint before {node!r} to replay.")
        fork_config: RunnableConfig = {
            "configurable": dict(fork.config["configurable"]),
            "callbacks": self._callbacks,
        }
        self._graph.invoke(None, fork_config)
        state = self._graph.get_state(self._config(thread_id))
        return TurnResult(
            next_question=None,
            closing=state.values.get("closing"),
            grade=state.values.get("grade"),
            coach=state.values.get("coach"),
            transcript=state.values.get("transcript", []),
        )

    @property
    def budget_callback(self) -> BudgetCallback:
        """The callback tracking token usage and budget across the session."""
        return self._budget_callback

    def session_usage(self) -> dict[str, ModelUsage]:
        """Return per-model token usage accumulated during the session.

        Returns:
            A mapping from model name to its recorded usage.
        """
        return self._budget_callback.usage

    def close(self) -> None:
        """Close the underlying checkpoint database connection"""
        self._conn.close()

    def __enter__(self) -> Self:
        """Enter the context manager, returning this engine."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the context manager, closing the engine"""
        self.close()

    def _config(self, thread_id: str) -> RunnableConfig:
        """Build the runnable config binding a thread id and the engine's callbacks"""
        return {"configurable": {"thread_id": thread_id}, "callbacks": self._callbacks}

    def _pending_session(self, thread_id: str) -> SessionSnapshot:
        """Snapshot a session, raising if it is already finished"""
        snapshot = self._snapshot(thread_id)
        if snapshot.finished:
            raise EngineError("The last session is already finished. Start a new one.")
        return snapshot

    def _snapshot(self, thread_id: str) -> SessionSnapshot:
        """Build a session snapshot from a thread's checkpointed graph state"""
        state = self._graph.get_state(self._config(thread_id))
        values = state.values
        if "profile" not in values:
            raise EngineError("This interview didn't get far enough to reopen.")
        needs_level = "ask_level" in state.next
        finished = not state.next
        role_context = values.get("role_context")
        return SessionSnapshot(
            thread_id=thread_id,
            profile=values["profile"],
            needs_level=needs_level,
            level=role_context.target_level if role_context is not None else None,
            question=(
                None
                if needs_level or finished
                else (values.get("redirect") or values.get("question"))
            ),
            transcript=list(values.get("transcript", [])),
            finished=finished,
            closing=values.get("closing"),
            grade=values.get("grade"),
            coach=values.get("coach"),
        )

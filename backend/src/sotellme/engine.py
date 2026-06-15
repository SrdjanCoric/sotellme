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
from sotellme.interviewer import Turn
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
    def decide(self, situation: DirectorSituation) -> DirectorDecision: ...


class Interviewer(Protocol):
    def question_for(
        self,
        decision: DirectorDecision,
        profile: CandidateProfile,
        context: RoleContext,
        brief: str,
        transcript: Sequence[Turn],
    ) -> str: ...

    def closing_turn(self, transcript: Sequence[Turn]) -> str: ...


class EngineError(Exception):
    pass


class InterviewState(TypedDict, total=False):
    cv_path: str
    cv_text: str
    posting_text: str
    profile: CandidateProfile
    role_context: RoleContext
    emphasis: list[str]
    company_brief: str
    question: str
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


class TurnResult(BaseModel):
    next_question: str | None = None
    closing: str | None = None
    grade: SessionGrade | None = None
    coach: CoachReport | None = None
    transcript: list[Turn] = []

    @property
    def finished(self) -> bool:
        return self.next_question is None


def _extract(state: InterviewState) -> InterviewState:
    return {"cv_text": extract_cv_text(Path(state["cv_path"]))}


def _ask_level(state: InterviewState) -> InterviewState:
    context = state["role_context"]
    if context.target_level is not None:
        return {}
    level = interrupt({"ask_level": True})
    return {"role_context": context.model_copy(update={"target_level": level})}


def _derive_emphasis(state: InterviewState) -> InterviewState:
    level = state["role_context"].target_level
    return {"emphasis": list(level_emphasis(level)) if level is not None else []}


def _await_answer(state: InterviewState) -> InterviewState:
    answer = interrupt({"question": state["question"]})
    turn = Turn(question=state["question"], answer=str(answer))
    return {"transcript": [*state.get("transcript", []), turn]}


class InterviewEngine:
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
        question_cap: int = DEFAULT_QUESTION_CAP,
        follow_up_cap: int = DEFAULT_FOLLOW_UP_CAP,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir
        self._callbacks = callbacks or []
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
            envelope = EnvelopeState(
                questions_asked=state.get("questions_asked", 0),
                consecutive_follow_ups=state.get("consecutive_follow_ups", 0),
            )
            if not question_allowed(envelope, question_cap):
                return {"decision": DirectorDecision(action="wrap_up", reason=ENVELOPE_WRAP_REASON)}
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
        graph.add_edge("await_answer", "assess")
        graph.add_edge("assess", "direct")
        graph.add_edge("pose_closing", "grade")
        graph.add_edge("grade", "coach")
        graph.add_edge("coach", END)
        serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINTED_TYPES)
        self._graph = graph.compile(checkpointer=SqliteSaver(self._conn, serde=serde))

    def start(self, cv_path: Path, posting_text: str | None = None) -> SessionSnapshot:
        thread_id = uuid.uuid4().hex
        initial: InterviewState = {"cv_path": str(cv_path)}
        if posting_text is not None:
            initial["posting_text"] = posting_text
        self._graph.invoke(initial, self._config(thread_id))
        (self._data_dir / "latest").write_text(thread_id)
        return self._pending_session(thread_id)

    def resume_latest(self) -> SessionSnapshot:
        return self._pending_session(self._latest_thread())

    def snapshot_latest(self) -> SessionSnapshot:
        return self._snapshot(self._latest_thread())

    def _latest_thread(self) -> str:
        latest = self._data_dir / "latest"
        if not latest.exists():
            raise EngineError("No session to resume. Start one with: sotellme interview")
        return latest.read_text().strip()

    def submit_level(self, thread_id: str, level: TargetLevel) -> SessionSnapshot:
        if level not in get_args(TargetLevel):
            valid = ", ".join(get_args(TargetLevel))
            raise EngineError(f"Unknown level {level!r}: choose one of {valid}.")
        self._graph.invoke(Command(resume=level), self._config(thread_id))
        return self._pending_session(thread_id)

    def submit_answer(self, thread_id: str, answer: str) -> TurnResult:
        self._graph.invoke(Command(resume=answer), self._config(thread_id))
        state = self._graph.get_state(self._config(thread_id))
        if state.next:
            return TurnResult(next_question=state.values["question"])
        return TurnResult(
            next_question=None,
            closing=state.values.get("closing"),
            grade=state.values.get("grade"),
            coach=state.values.get("coach"),
            transcript=state.values.get("transcript", []),
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}, "callbacks": self._callbacks}

    def _pending_session(self, thread_id: str) -> SessionSnapshot:
        snapshot = self._snapshot(thread_id)
        if snapshot.finished:
            raise EngineError("The last session is already finished. Start a new one.")
        return snapshot

    def _snapshot(self, thread_id: str) -> SessionSnapshot:
        state = self._graph.get_state(self._config(thread_id))
        values = state.values
        needs_level = "ask_level" in state.next
        finished = not state.next
        role_context = values.get("role_context")
        return SessionSnapshot(
            thread_id=thread_id,
            profile=values["profile"],
            needs_level=needs_level,
            level=role_context.target_level if role_context is not None else None,
            question=None if needs_level or finished else values.get("question"),
            transcript=list(values.get("transcript", [])),
            finished=finished,
            closing=values.get("closing"),
            grade=values.get("grade"),
            coach=values.get("coach"),
        )

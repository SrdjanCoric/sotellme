import sqlite3
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, TypedDict, cast, get_args

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from sotellme.coverage import (
    DEFAULT_FOLLOWUP_CAP,
    DEFAULT_MAX_COMPETENCIES,
    MOTIVATION_TOPICS,
    CoverageState,
    Gap,
    Motivation,
    MotivationTopic,
    NextCompetency,
    Probe,
    StarFlags,
    next_action,
    plan_competencies,
)
from sotellme.extraction import extract_cv_text
from sotellme.interviewer import Turn
from sotellme.profile import CandidateProfile, Project, Role
from sotellme.role import CompetencyWeight, RoleContext, TargetLevel, default_role_context

CHECKPOINTED_TYPES = (CandidateProfile, Role, Project, Turn, RoleContext, CompetencyWeight)

ProfileParser = Callable[[str], CandidateProfile]
StarFlagger = Callable[[str], StarFlags]
RoleBuilder = Callable[[str], RoleContext]


class Interviewer(Protocol):
    def competency_question(
        self, profile: CandidateProfile, transcript: Sequence[Turn], competency: str
    ) -> str: ...

    def probe_question(
        self, profile: CandidateProfile, transcript: Sequence[Turn], gaps: tuple[Gap, ...]
    ) -> str: ...

    def motivation_question(
        self,
        context: RoleContext,
        posting_text: str,
        transcript: Sequence[Turn],
        topic: MotivationTopic,
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
    competency_plan: list[str]
    competency_index: int
    motivation_topics: list[str]
    motivation_asked: int
    question: str
    transcript: list[Turn]
    followups_used: int
    gaps: list[Gap]
    decision: str
    closing: str


class SessionHandle(BaseModel):
    thread_id: str
    question: str | None
    needs_level: bool
    profile: CandidateProfile


class TurnResult(BaseModel):
    next_question: str | None = None
    closing: str | None = None

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


def _await_answer(state: InterviewState) -> InterviewState:
    answer = interrupt({"question": state["question"]})
    turn = Turn(question=state["question"], answer=str(answer))
    return {"transcript": [*state.get("transcript", []), turn]}


class InterviewEngine:
    def __init__(
        self,
        data_dir: Path,
        profile_parser: ProfileParser,
        star_flagger: StarFlagger,
        interviewer: Interviewer,
        role_builder: RoleBuilder,
        followup_cap: int = DEFAULT_FOLLOWUP_CAP,
        max_competencies: int = DEFAULT_MAX_COMPETENCIES,
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
            return {
                "role_context": context,
                "competency_plan": list(plan_competencies(context, max_competencies)),
                "competency_index": 0,
                "motivation_topics": list(MOTIVATION_TOPICS) if posting else [],
                "motivation_asked": 0,
            }

        def pose_competency(state: InterviewState) -> InterviewState:
            competency = state["competency_plan"][state.get("competency_index", 0)]
            question = interviewer.competency_question(
                state["profile"], state.get("transcript", []), competency
            )
            return {"question": question, "followups_used": 0}

        def assess(state: InterviewState) -> InterviewState:
            in_motivation = state.get("motivation_asked", 0) > 0
            index = state.get("competency_index", 0)
            action = next_action(
                CoverageState(
                    flags=None if in_motivation else star_flagger(state["transcript"][-1].answer),
                    followups_used=state.get("followups_used", 0),
                    competencies_remaining=()
                    if in_motivation
                    else tuple(state["competency_plan"][index + 1 :]),
                    motivation_remaining=tuple(
                        cast(
                            list[MotivationTopic],
                            state.get("motivation_topics", []),
                        )[state.get("motivation_asked", 0) :]
                    ),
                    in_motivation=in_motivation,
                ),
                followup_cap=followup_cap,
            )
            if isinstance(action, Probe):
                return {"decision": "probe", "gaps": list(action.gaps)}
            if isinstance(action, NextCompetency):
                return {"decision": "next_competency", "competency_index": index + 1}
            if isinstance(action, Motivation):
                return {"decision": "motivation"}
            return {"decision": "finish"}

        def pose_probe(state: InterviewState) -> InterviewState:
            question = interviewer.probe_question(
                state["profile"], state["transcript"], tuple(state["gaps"])
            )
            return {"question": question, "followups_used": state.get("followups_used", 0) + 1}

        def pose_motivation(state: InterviewState) -> InterviewState:
            asked = state.get("motivation_asked", 0)
            topic = cast(list[MotivationTopic], state["motivation_topics"])[asked]
            question = interviewer.motivation_question(
                state["role_context"],
                state.get("posting_text", ""),
                state["transcript"],
                topic,
            )
            return {"question": question, "motivation_asked": asked + 1}

        def pose_closing(state: InterviewState) -> InterviewState:
            return {"closing": interviewer.closing_turn(state["transcript"])}

        def route_after_assess(state: InterviewState) -> str:
            return {
                "probe": "pose_probe",
                "next_competency": "pose_competency",
                "motivation": "pose_motivation",
            }.get(state["decision"], "pose_closing")

        graph = StateGraph(InterviewState)
        graph.add_node("extract", _extract)
        graph.add_node("parse_profile", parse_profile)
        graph.add_node("build_context", build_context)
        graph.add_node("ask_level", _ask_level)
        graph.add_node("pose_competency", pose_competency)
        graph.add_node("await_answer", _await_answer)
        graph.add_node("assess", assess)
        graph.add_node("pose_probe", pose_probe)
        graph.add_node("pose_motivation", pose_motivation)
        graph.add_node("pose_closing", pose_closing)
        graph.add_edge(START, "extract")
        graph.add_edge("extract", "parse_profile")
        graph.add_edge("parse_profile", "build_context")
        graph.add_edge("build_context", "ask_level")
        graph.add_edge("ask_level", "pose_competency")
        graph.add_edge("pose_competency", "await_answer")
        graph.add_edge("await_answer", "assess")
        graph.add_conditional_edges(
            "assess",
            route_after_assess,
            ["pose_probe", "pose_competency", "pose_motivation", "pose_closing"],
        )
        graph.add_edge("pose_probe", "await_answer")
        graph.add_edge("pose_motivation", "await_answer")
        graph.add_edge("pose_closing", END)
        serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINTED_TYPES)
        self._graph = graph.compile(checkpointer=SqliteSaver(self._conn, serde=serde))

    def start(self, cv_path: Path, posting_text: str | None = None) -> SessionHandle:
        thread_id = uuid.uuid4().hex
        initial: InterviewState = {"cv_path": str(cv_path)}
        if posting_text is not None:
            initial["posting_text"] = posting_text
        self._graph.invoke(initial, self._config(thread_id))
        (self._data_dir / "latest").write_text(thread_id)
        return self._pending_session(thread_id)

    def resume_latest(self) -> SessionHandle:
        latest = self._data_dir / "latest"
        if not latest.exists():
            raise EngineError("No session to resume. Start one with: sotellme interview")
        return self._pending_session(latest.read_text().strip())

    def submit_level(self, thread_id: str, level: TargetLevel) -> SessionHandle:
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
        return TurnResult(next_question=None, closing=state.values.get("closing"))

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

    def _pending_session(self, thread_id: str) -> SessionHandle:
        state = self._graph.get_state(self._config(thread_id))
        if not state.next:
            raise EngineError("The last session is already finished. Start a new one.")
        needs_level = "ask_level" in state.next
        return SessionHandle(
            thread_id=thread_id,
            question=None if needs_level else state.values["question"],
            needs_level=needs_level,
            profile=state.values["profile"],
        )

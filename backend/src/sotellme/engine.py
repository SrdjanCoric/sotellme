import sqlite3
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from sotellme.coverage import (
    DEFAULT_FOLLOWUP_CAP,
    CoverageState,
    Gap,
    Probe,
    StarFlags,
    next_action,
)
from sotellme.extraction import extract_cv_text
from sotellme.interviewer import Turn
from sotellme.profile import CandidateProfile, Project, Role

CHECKPOINTED_TYPES = (CandidateProfile, Role, Project, Turn)

ProfileParser = Callable[[str], CandidateProfile]
StarFlagger = Callable[[str], StarFlags]


class Interviewer(Protocol):
    def opening_question(self, profile: CandidateProfile) -> str: ...

    def probe_question(
        self, profile: CandidateProfile, transcript: Sequence[Turn], gaps: tuple[Gap, ...]
    ) -> str: ...

    def closing_turn(self, transcript: Sequence[Turn]) -> str: ...


class EngineError(Exception):
    pass


class InterviewState(TypedDict, total=False):
    cv_path: str
    cv_text: str
    profile: CandidateProfile
    question: str
    transcript: list[Turn]
    followups_used: int
    gaps: list[Gap]
    decision: str
    closing: str


class SessionHandle(BaseModel):
    thread_id: str
    question: str
    profile: CandidateProfile


class TurnResult(BaseModel):
    next_question: str | None = None
    closing: str | None = None

    @property
    def finished(self) -> bool:
        return self.next_question is None


def _extract(state: InterviewState) -> InterviewState:
    return {"cv_text": extract_cv_text(Path(state["cv_path"]))}


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
        followup_cap: int = DEFAULT_FOLLOWUP_CAP,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir
        self._callbacks = callbacks or []
        self._conn = sqlite3.connect(data_dir / "checkpoints.sqlite", check_same_thread=False)

        def parse_profile(state: InterviewState) -> InterviewState:
            return {"profile": profile_parser(state["cv_text"])}

        def pose_opening(state: InterviewState) -> InterviewState:
            return {"question": interviewer.opening_question(state["profile"])}

        def assess(state: InterviewState) -> InterviewState:
            flags = star_flagger(state["transcript"][-1].answer)
            action = next_action(
                CoverageState(flags=flags, followups_used=state.get("followups_used", 0)),
                followup_cap=followup_cap,
            )
            if isinstance(action, Probe):
                return {"decision": "probe", "gaps": list(action.gaps)}
            return {"decision": "finish"}

        def pose_probe(state: InterviewState) -> InterviewState:
            question = interviewer.probe_question(
                state["profile"], state["transcript"], tuple(state["gaps"])
            )
            return {"question": question, "followups_used": state.get("followups_used", 0) + 1}

        def pose_closing(state: InterviewState) -> InterviewState:
            return {"closing": interviewer.closing_turn(state["transcript"])}

        def route_after_assess(state: InterviewState) -> str:
            return "pose_probe" if state["decision"] == "probe" else "pose_closing"

        graph = StateGraph(InterviewState)
        graph.add_node("extract", _extract)
        graph.add_node("parse_profile", parse_profile)
        graph.add_node("pose_opening", pose_opening)
        graph.add_node("await_answer", _await_answer)
        graph.add_node("assess", assess)
        graph.add_node("pose_probe", pose_probe)
        graph.add_node("pose_closing", pose_closing)
        graph.add_edge(START, "extract")
        graph.add_edge("extract", "parse_profile")
        graph.add_edge("parse_profile", "pose_opening")
        graph.add_edge("pose_opening", "await_answer")
        graph.add_edge("await_answer", "assess")
        graph.add_conditional_edges("assess", route_after_assess, ["pose_probe", "pose_closing"])
        graph.add_edge("pose_probe", "await_answer")
        graph.add_edge("pose_closing", END)
        serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINTED_TYPES)
        self._graph = graph.compile(checkpointer=SqliteSaver(self._conn, serde=serde))

    def start(self, cv_path: Path) -> SessionHandle:
        thread_id = uuid.uuid4().hex
        self._graph.invoke({"cv_path": str(cv_path)}, self._config(thread_id))
        (self._data_dir / "latest").write_text(thread_id)
        return self._pending_session(thread_id)

    def resume_latest(self) -> SessionHandle:
        latest = self._data_dir / "latest"
        if not latest.exists():
            raise EngineError("No session to resume. Start one with: sotellme interview")
        return self._pending_session(latest.read_text().strip())

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
        return SessionHandle(
            thread_id=thread_id,
            question=state.values["question"],
            profile=state.values["profile"],
        )

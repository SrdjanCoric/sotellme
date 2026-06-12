import sqlite3
import uuid
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Self, TypedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from sotellme.extraction import extract_cv_text
from sotellme.profile import CandidateProfile
from sotellme.prompts import FIXED_OPENING_QUESTION

ProfileParser = Callable[[str], CandidateProfile]


class EngineError(Exception):
    pass


class InterviewState(TypedDict, total=False):
    cv_path: str
    cv_text: str
    profile: CandidateProfile
    question: str
    answer: str


class SessionHandle(BaseModel):
    thread_id: str
    question: str
    profile: CandidateProfile


def _extract(state: InterviewState) -> InterviewState:
    return {"cv_text": extract_cv_text(Path(state["cv_path"]))}


def _pose_question(state: InterviewState) -> InterviewState:
    return {"question": FIXED_OPENING_QUESTION}


def _await_answer(state: InterviewState) -> InterviewState:
    answer = interrupt({"question": state["question"]})
    return {"answer": str(answer)}


class InterviewEngine:
    def __init__(
        self,
        data_dir: Path,
        profile_parser: ProfileParser,
        callbacks: list[BaseCallbackHandler] | None = None,
    ) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir
        self._callbacks = callbacks or []
        self._conn = sqlite3.connect(data_dir / "checkpoints.sqlite", check_same_thread=False)

        def parse_profile(state: InterviewState) -> InterviewState:
            return {"profile": profile_parser(state["cv_text"])}

        graph = StateGraph(InterviewState)
        graph.add_node("extract", _extract)
        graph.add_node("parse_profile", parse_profile)
        graph.add_node("pose_question", _pose_question)
        graph.add_node("await_answer", _await_answer)
        graph.add_edge(START, "extract")
        graph.add_edge("extract", "parse_profile")
        graph.add_edge("parse_profile", "pose_question")
        graph.add_edge("pose_question", "await_answer")
        graph.add_edge("await_answer", END)
        self._graph = graph.compile(checkpointer=SqliteSaver(self._conn))

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

    def submit_answer(self, thread_id: str, answer: str) -> str:
        final = self._graph.invoke(Command(resume=answer), self._config(thread_id))
        return str(final["answer"])

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

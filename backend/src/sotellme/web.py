import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler

from sotellme.assessor import AssessorError
from sotellme.catalog import Catalog, CatalogError, load_catalog
from sotellme.cli import (
    NO_COACHING_MESSAGE,
    TARGET_LEVELS,
    _data_dir,
    build_engine,
    format_score_summary,
)
from sotellme.coach import CoachingError, CoachReport
from sotellme.config import (
    AGENT_ROLES,
    AGENT_TAG_PREFIX,
    PROVIDER_KEY_VARS,
    AgentModel,
    ModelConfigError,
    resolve_model_config,
)
from sotellme.director import DirectorError
from sotellme.engine import EngineError, InterviewEngine, SessionSnapshot, TurnResult
from sotellme.grader import GradingError, SessionGrade
from sotellme.interviewer import Turn
from sotellme.posting import PostingInputError, resolve_posting_text
from sotellme.profile import CandidateProfile
from sotellme.report import render_report, write_report
from sotellme.role import TargetLevel
from sotellme.tracing import TracingError, langfuse_callbacks

DEFAULT_CHOICE = "— provider default —"

LINK_MODE = "Link"
TEXT_MODE = "Paste text"

TURN_ERRORS = (AssessorError, DirectorError, GradingError, CoachingError, EngineError)

DEFAULT_TEST_ANSWER = (
    "At my last role I led the migration of our billing service to a new datastore. "
    "I scoped the work into phases, coordinated three engineers, and we cut p99 latency "
    "by 40% with zero downtime. The hardest part was backfilling without double-charging, "
    "which I solved with idempotency keys and a reconciliation job."
)

Phase = Literal["setup", "level", "interview", "report"]

ChatMessage = tuple[Literal["assistant", "user"], str]


@dataclass
class WebState:
    thread_id: str
    profile: CandidateProfile
    needs_level: bool
    question: str | None
    level: TargetLevel | None = None
    answered: list[Turn] = field(default_factory=list)
    finished: bool = False
    closing: str | None = None
    grade: SessionGrade | None = None
    coach: CoachReport | None = None
    transcript: list[Turn] = field(default_factory=list)


def state_from_snapshot(snapshot: SessionSnapshot) -> WebState:
    return WebState(
        thread_id=snapshot.thread_id,
        profile=snapshot.profile,
        needs_level=snapshot.needs_level,
        question=snapshot.question,
        level=snapshot.level,
        answered=list(snapshot.transcript),
        finished=snapshot.finished,
        closing=snapshot.closing,
        grade=snapshot.grade,
        coach=snapshot.coach,
        transcript=list(snapshot.transcript),
    )


def state_after_answer(state: WebState, answer: str, result: TurnResult) -> WebState:
    answered = state.answered
    if state.question is not None:
        answered = [*answered, Turn(question=state.question, answer=answer)]
    return replace(
        state,
        needs_level=False,
        answered=answered,
        question=result.next_question,
        finished=result.finished,
        closing=result.closing,
        grade=result.grade,
        coach=result.coach,
        transcript=list(result.transcript) if result.finished else answered,
    )


def phase_of(state: WebState | None) -> Phase:
    if state is None:
        return "setup"
    if state.needs_level:
        return "level"
    if state.question is not None and not state.finished:
        return "interview"
    return "report"


def model_choices(catalog: Catalog) -> list[str]:
    return [
        f"{provider}:{model}"
        for provider in sorted(catalog.providers)
        for model in catalog.providers[provider].models
    ]


def agent_overrides_from_selections(selections: Mapping[str, str]) -> dict[str, AgentModel]:
    overrides: dict[str, AgentModel] = {}
    for role, choice in selections.items():
        if choice == DEFAULT_CHOICE:
            continue
        provider, _, model = choice.partition(":")
        overrides[role] = AgentModel(provider=provider, model=model)
    return overrides


def default_provider(catalog: Catalog, env: Mapping[str, str]) -> str | None:
    chosen = env.get("SOTELLME_PROVIDER")
    if chosen in catalog.providers:
        return chosen
    for provider in sorted(catalog.providers):
        if env.get(PROVIDER_KEY_VARS.get(provider, "")):
            return provider
    return None


def clean_posting(raw: str) -> str | None:
    stripped = raw.strip()
    return stripped or None


def posting_to_resolve(mode: str, url: str, text: str) -> tuple[str | None, bool]:
    if mode == LINK_MODE:
        return clean_posting(url), True
    return clean_posting(text), False


def save_upload(name: str, data: bytes, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(suffix=Path(name).suffix, dir=directory)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return Path(raw_path)


def chat_messages(state: WebState) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for turn in state.answered:
        messages.append(("assistant", turn.question))
        messages.append(("user", turn.answer))
    if state.question is not None:
        messages.append(("assistant", state.question))
    elif state.closing:
        messages.append(("assistant", state.closing))
    return messages


AGENT_STEP_LABELS = {
    "parser": "Reading your CV",
    "role_builder": "Sizing up the role",
    "researcher": "Researching the company",
    "director": "Choosing the next question",
    "interviewer": "Writing the question",
    "assessor": "Weighing your answer",
    "guardrail": "Checking your reply",
    "grader": "Grading your answers",
    "coach": "Writing your coaching",
}


def agent_step_label(tags: Any) -> str | None:
    if not tags:
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(AGENT_TAG_PREFIX):
            return AGENT_STEP_LABELS.get(tag[len(AGENT_TAG_PREFIX) :])
    return None


class _ModelProgress(BaseCallbackHandler):
    """Ticks an `st.status` label and writes a line per step so a long wait shows its work."""

    def __init__(self) -> None:
        self._status: Any = None
        self._label = ""
        self._step = 0
        self._last_written: str | None = None

    def aim(self, status: Any, label: str) -> None:
        self._status = status
        self._label = label
        self._step = 0
        self._last_written = None

    def clear(self) -> None:
        self._status = None

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        if self._status is None:
            return
        self._step += 1
        try:
            self._status.update(label=f"{self._label} ({self._step})")
            step_label = agent_step_label(kwargs.get("tags"))
            if step_label is not None and step_label != self._last_written:
                self._status.write(step_label)
                self._last_written = step_label
        except Exception:
            self._status = None


def run() -> None:
    import streamlit as st

    st.set_page_config(page_title="sotellme", page_icon="🗣️")
    st.title("sotellme")
    st.caption("A mock behavioral interview, built from your CV and the job you're chasing.")

    try:
        catalog = load_catalog(_data_dir())
    except CatalogError as exc:
        st.error(str(exc))
        return

    state = _recover(catalog)
    _render_sidebar(state)
    if state is None:
        _render_setup(catalog)
        return
    phase = phase_of(state)
    if phase == "report":
        _render_report(state)
        return
    engine = st.session_state.get("engine")
    if not isinstance(engine, InterviewEngine):
        st.session_state.pop("state", None)
        _render_setup(catalog)
        return
    if phase == "level":
        _render_level(engine, state)
    else:
        _render_interview(engine, state)


def _tracing_callbacks() -> list[BaseCallbackHandler]:
    import streamlit as st

    try:
        return langfuse_callbacks(os.environ)
    except TracingError as exc:
        st.error(str(exc))
        st.stop()
        raise  # unreachable: st.stop() halts the run


def _recovery_engine(catalog: Catalog) -> InterviewEngine | None:
    provider = default_provider(catalog, os.environ)
    if provider is None:
        return None
    try:
        config = resolve_model_config(env=os.environ, provider=provider, catalog=catalog)
    except ModelConfigError:
        return None
    return build_engine(config, _tracing_callbacks())


def _recover(catalog: Catalog) -> WebState | None:
    import streamlit as st

    existing = st.session_state.get("state")
    if isinstance(existing, WebState):
        return existing
    if st.session_state.get("new_interview"):
        return None
    engine = _recovery_engine(catalog)
    if engine is None:
        return None
    try:
        snapshot = engine.snapshot_latest()
    except EngineError:
        return None
    st.session_state.engine = engine
    state = state_from_snapshot(snapshot)
    st.session_state.state = state
    return state


def _render_sidebar(state: WebState | None) -> None:
    import streamlit as st

    with st.sidebar:
        st.subheader("Menu")
        if state is not None:
            if st.button("New interview", use_container_width=True):
                for key in ("state", "engine", "report_path"):
                    st.session_state.pop(key, None)
                st.session_state.new_interview = True
                st.rerun()
        else:
            st.caption("Start an interview to see options here.")


def _render_setup(catalog: Catalog) -> None:
    import streamlit as st

    st.subheader("Start an interview")

    st.write("Job posting (optional)")
    posting_mode = st.radio(
        "Posting source",
        (LINK_MODE, TEXT_MODE),
        horizontal=True,
        label_visibility="collapsed",
    )

    providers = sorted(catalog.providers)
    preferred = default_provider(catalog, os.environ)
    selections: dict[str, str] = {}
    with st.form("setup", border=False):
        cv_file = st.file_uploader("Your CV", type=["pdf", "md", "markdown", "txt"])
        posting_url = ""
        posting_text = ""
        if posting_mode == LINK_MODE:
            posting_url = st.text_input("Posting link", placeholder="https://…")
        else:
            posting_text = st.text_area("Posting text", placeholder="Paste the posting text…")
        provider = st.selectbox(
            "Provider",
            providers,
            index=providers.index(preferred) if preferred in providers else 0,
        )
        with st.expander("Advanced: choose a model per step"):
            st.caption("Leave a step on its provider default, or pin it to a catalog model.")
            choices = [DEFAULT_CHOICE, *model_choices(catalog)]
            for role in AGENT_ROLES:
                selections[role] = st.selectbox(
                    role.replace("_", " "), choices, key=f"agent_{role}"
                )
        submitted = st.form_submit_button("Start interview", type="primary")

    if not submitted:
        return
    if cv_file is None:
        st.error("Upload your CV to start.")
        return
    try:
        config = resolve_model_config(
            env=os.environ,
            provider=provider,
            catalog=catalog,
            agent_overrides=agent_overrides_from_selections(selections),
        )
    except ModelConfigError as exc:
        st.error(str(exc))
        return
    value, needs_fetch = posting_to_resolve(posting_mode, posting_url, posting_text)
    resolved_posting: str | None = value
    if value is not None and needs_fetch:
        try:
            with st.spinner("Reading the posting…"):
                resolved_posting = resolve_posting_text(value)
        except PostingInputError as exc:
            st.error(str(exc))
            return
    progress = _ModelProgress()
    engine = build_engine(config, [*_tracing_callbacks(), progress])
    st.session_state.progress = progress
    cv_path = save_upload(cv_file.name, cv_file.getvalue(), _upload_dir())
    with st.status("Reading your CV and researching the role…") as status:
        progress.aim(status, "Reading your CV and researching the role")
        snapshot = engine.start(cv_path, posting_text=resolved_posting)
        progress.clear()
    st.session_state.engine = engine
    st.session_state.state = state_from_snapshot(snapshot)
    st.session_state.pop("new_interview", None)
    st.rerun()


def _render_level(engine: InterviewEngine, state: WebState) -> None:
    import streamlit as st

    st.info("This posting doesn't state a level. What level is this interview for?")
    level = st.selectbox("Target level", TARGET_LEVELS, format_func=str.capitalize)
    if st.button("Continue", type="primary"):
        with st.status("Setting up the interview…") as status:
            _aim_progress(status, "Setting up the interview")
            snapshot = engine.submit_level(state.thread_id, level)
        st.session_state.state = state_from_snapshot(snapshot)
        st.rerun()


def _render_level_caption(state: WebState) -> None:
    import streamlit as st

    if state.level is not None:
        st.caption(f"Interviewing at the {state.level} level.")


def _render_interview(engine: InterviewEngine, state: WebState) -> None:
    import streamlit as st

    _render_level_caption(state)
    for role, content in chat_messages(state):
        st.chat_message(role).write(content)
    _render_test_autopilot(engine, state)
    answer = st.chat_input("Your answer")
    if answer:
        st.chat_message("user").write(answer)
        with st.status("Thinking…") as status:
            _aim_progress(status, "Thinking")
            try:
                result = engine.submit_answer(state.thread_id, answer)
            except TURN_ERRORS as exc:
                status.update(state="error")
                st.error(str(exc))
                return
        st.session_state.state = state_after_answer(state, answer, result)
        st.rerun()


def _test_answer() -> str | None:
    if not os.environ.get("SOTELLME_TEST_MODE"):
        return None
    return os.environ.get("SOTELLME_TEST_ANSWER") or DEFAULT_TEST_ANSWER


def _render_test_autopilot(engine: InterviewEngine, state: WebState) -> None:
    import streamlit as st

    answer = _test_answer()
    if answer is None:
        return
    if not st.button("⏩ Finish with test answers", help="Test mode: auto-answer to the end."):
        return
    current = state
    with st.status("Auto-answering to the end…") as status:
        _aim_progress(status, "Auto-answering")
        while phase_of(current) == "interview":
            try:
                result = engine.submit_answer(current.thread_id, answer)
            except TURN_ERRORS as exc:
                status.update(state="error")
                st.error(str(exc))
                st.session_state.state = current
                return
            current = state_after_answer(current, answer, result)
    st.session_state.state = current
    st.rerun()


def _aim_progress(status: Any, label: str) -> None:
    import streamlit as st

    progress = st.session_state.get("progress")
    if isinstance(progress, _ModelProgress):
        progress.aim(status, label)


def _render_report(state: WebState) -> None:
    import streamlit as st

    _render_level_caption(state)
    for role, content in chat_messages(state):
        st.chat_message(role).write(content)
    grade = state.grade
    coach = state.coach
    if coach is not None and grade is not None and grade.scores:
        st.subheader("Scorecard")
        st.text(format_score_summary(grade))
        st.markdown(render_report(coach, state.transcript))
        report_path = st.session_state.get("report_path")
        if report_path is None and st.button("Save report"):
            report_path = write_report(coach, state.transcript, Path.cwd(), datetime.now())
            st.session_state.report_path = report_path
        if report_path is not None:
            st.caption(f"Saved your full report to {report_path}")
    else:
        st.info(NO_COACHING_MESSAGE)


def _upload_dir() -> Path:
    return Path(tempfile.gettempdir())


if __name__ == "__main__":
    run()

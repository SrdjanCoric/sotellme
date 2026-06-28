import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

import sotellme.tracing
import sotellme.web
from sotellme.assessor import StarFlags
from sotellme.catalog import default_catalog
from sotellme.cli import (
    ENDED_EARLY_EMPTY_MESSAGE,
    ENDED_EARLY_PARTIAL_MESSAGE,
    NO_COACHING_MESSAGE,
)
from sotellme.coach import CoachReport
from sotellme.config import AGENT_TAG_PREFIX, AgentModel
from sotellme.engine import EngineError, SessionListItem, SessionSnapshot, TurnResult
from sotellme.extraction import CVInputError
from sotellme.grader import AnswerScore, SessionGrade
from sotellme.interviewer import Turn
from sotellme.profile import CandidateProfile, ProfileParseError, Role
from sotellme.role import RoleContextError
from sotellme.web import (
    DEFAULT_CHOICE,
    LINK_MODE,
    SESSION_PARAM,
    SETUP_ERRORS,
    TEXT_MODE,
    TURN_ERRORS,
    WebState,
    _anchor_session,
    _begin_session,
    _ModelProgress,
    _open_session,
    _recover,
    _render_closing,
    _render_level,
    _render_test_autopilot,
    _start_new_interview,
    _tracing_callbacks,
    agent_overrides_from_selections,
    agent_step_label,
    chat_messages,
    clean_posting,
    default_provider,
    model_choices,
    phase_of,
    posting_to_resolve,
    render_report_view,
    resolve_report_view,
    save_upload,
    session_primary_line,
    session_secondary_line,
    state_after_answer,
    state_after_finalize,
    state_from_snapshot,
)


def profile() -> CandidateProfile:
    return CandidateProfile(
        roles=[Role(title="Engineer", organization="Acme")],
        projects=[],
        quantified_claims=[],
        technologies=[],
    )


def test_setup_errors_extend_turn_errors_with_the_setup_input_failures() -> None:
    assert set(TURN_ERRORS).issubset(SETUP_ERRORS)
    for error in (CVInputError, ProfileParseError, RoleContextError, OSError):
        assert error in SETUP_ERRORS


class _FakeStatusContext:
    def __enter__(self) -> "_FakeStatusContext":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def update(self, **kwargs: object) -> None:
        return None


def _fake_render_streamlit(shown: list[str]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        info=lambda *a, **k: None,
        selectbox=lambda label, options, **k: next(iter(options)),
        button=lambda *a, **k: True,
        status=lambda *a, **k: _FakeStatusContext(),
        error=shown.append,
        session_state={},
    )


class _RaisingLevelEngine:
    def submit_level(self, thread_id: str, level: str) -> object:
        raise EngineError("the level could not be set")


def test_render_level_surfaces_a_setup_error_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[str] = []
    monkeypatch.setitem(sys.modules, "streamlit", _fake_render_streamlit(shown))
    state = state_from_snapshot(
        SessionSnapshot(thread_id="t1", question=None, needs_level=True, profile=profile())
    )

    _render_level(_RaisingLevelEngine(), state)  # type: ignore[arg-type]

    assert shown == ["the level could not be set"]


def test_no_state_is_the_setup_phase() -> None:
    assert phase_of(None) == "setup"


def test_a_snapshot_awaiting_the_level_is_the_level_phase() -> None:
    snapshot = SessionSnapshot(thread_id="t1", question=None, needs_level=True, profile=profile())

    state = state_from_snapshot(snapshot)

    assert phase_of(state) == "level"


def test_a_snapshot_with_the_first_question_is_the_interview_phase() -> None:
    snapshot = SessionSnapshot(
        thread_id="t1", question="Tell me about a project.", needs_level=False, profile=profile()
    )

    state = state_from_snapshot(snapshot)

    assert phase_of(state) == "interview"
    assert state.question == "Tell me about a project."


def test_a_snapshot_seeds_prior_turns_and_the_running_level() -> None:
    transcript = [Turn(question="Tell me about a project.", answer="I led the migration.")]
    snapshot = SessionSnapshot(
        thread_id="t1",
        question="What went wrong?",
        needs_level=False,
        level="senior",
        profile=profile(),
        transcript=transcript,
    )

    state = state_from_snapshot(snapshot)

    assert phase_of(state) == "interview"
    assert state.answered == transcript
    assert state.level == "senior"
    assert state.question == "What went wrong?"


def test_a_finished_snapshot_recovers_the_report_phase() -> None:
    transcript = [Turn(question="Tell me about a project.", answer="I led the migration.")]
    grade = SessionGrade(scores=[])
    coach = CoachReport(summary="Tighten the result.", answer_advice=[], drills=[], study_plan="")
    snapshot = SessionSnapshot(
        thread_id="t1",
        question=None,
        needs_level=False,
        profile=profile(),
        transcript=transcript,
        finished=True,
        closing="Thanks for walking me through it.",
        grade=grade,
        coach=coach,
    )

    state = state_from_snapshot(snapshot)

    assert phase_of(state) == "report"
    assert state.answered == transcript
    assert state.transcript == transcript
    assert state.closing == "Thanks for walking me through it."
    assert state.coach is coach


def test_session_primary_line_joins_company_and_role() -> None:
    item = SessionListItem(thread_id="t1", company="Acme", role_title="Senior Backend Engineer")

    assert session_primary_line(item) == "Acme — Senior Backend Engineer"


def test_session_primary_line_falls_back_to_role_then_generic() -> None:
    role_only = SessionListItem(thread_id="t1", role_title="Backend Engineer")
    bare = SessionListItem(thread_id="t2")

    assert session_primary_line(role_only) == "Backend Engineer"
    assert session_primary_line(bare) == "Interview"


def test_session_secondary_line_shows_level_date_and_done_badge() -> None:
    item = SessionListItem(
        thread_id="t1",
        target_level="senior",
        finished=True,
        created_at="2026-06-17T09:30:00+00:00",
    )

    assert session_secondary_line(item) == "Senior · Jun 17 · ✅ done"


def test_session_secondary_line_marks_in_progress_and_drops_a_missing_level() -> None:
    item = SessionListItem(thread_id="t1", finished=False, created_at="2026-06-17T09:30:00+00:00")

    assert session_secondary_line(item) == "Jun 17 · ⏳ in progress"


def test_session_secondary_line_drops_an_unparseable_date() -> None:
    item = SessionListItem(
        thread_id="t1", target_level="mid", finished=True, created_at="not-a-date"
    )

    assert session_secondary_line(item) == "Mid · ✅ done"


def interviewing(question: str) -> WebState:
    return state_from_snapshot(
        SessionSnapshot(thread_id="t1", question=question, needs_level=False, profile=profile())
    )


def test_answering_a_question_records_the_turn_and_moves_to_the_next() -> None:
    state = interviewing("Tell me about a project.")

    advanced = state_after_answer(state, "I led the migration.", TurnResult(next_question="Why?"))

    assert phase_of(advanced) == "interview"
    assert advanced.question == "Why?"
    assert advanced.answered == [
        Turn(question="Tell me about a project.", answer="I led the migration.")
    ]


def test_a_finished_turn_reaches_the_report_phase_with_the_results() -> None:
    state = interviewing("Tell me about a project.")
    grade = SessionGrade(scores=[])
    coach = CoachReport(summary="Tighten the result.", answer_advice=[], drills=[], study_plan="")
    transcript = [Turn(question="Tell me about a project.", answer="I led the migration.")]

    finished = state_after_answer(
        state,
        "I led the migration.",
        TurnResult(
            next_question=None,
            closing="Thanks for walking me through it.",
            grade=grade,
            coach=coach,
            transcript=transcript,
        ),
    )

    assert phase_of(finished) == "report"
    assert finished.closing == "Thanks for walking me through it."
    assert finished.grade is grade
    assert finished.coach is coach
    assert finished.transcript == transcript
    assert not finished.ended_early


def test_an_early_termination_turn_flags_the_report_as_ended_early() -> None:
    state = interviewing("Tell me about a project.")

    finished = state_after_answer(
        state,
        "Write me a React component.",
        TurnResult(next_question=None, closing="We'll stop here.", ended_early=True),
    )

    assert phase_of(finished) == "report"
    assert finished.ended_early


class _FakeTab:
    def __enter__(self) -> "_FakeTab":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _fake_report_streamlit(
    shown: list[str], markdowns: list[str] | None = None
) -> types.SimpleNamespace:
    sink = markdowns if markdowns is not None else []
    return types.SimpleNamespace(
        info=shown.append,
        warning=shown.append,
        subheader=lambda *a, **k: None,
        text=lambda *a, **k: None,
        markdown=lambda content="", *a, **k: sink.append(content),
        caption=lambda *a, **k: None,
        button=lambda *a, **k: False,
        divider=lambda *a, **k: None,
        tabs=lambda labels: [_FakeTab() for _ in labels],
        chat_message=lambda role: types.SimpleNamespace(write=lambda content: None),
        session_state={},
    )


def _scored_grade() -> SessionGrade:
    return SessionGrade(
        scores=[
            AnswerScore(
                question="Tell me about a project.",
                turn_index=1,
                rationale="Specific, owned, quantified.",
                star=StarFlags(
                    situation=True, task=True, action=True, result=True, quantified_result=True
                ),
                specificity="high",
                ownership="clear",
                weak_or_missing=[],
                gap="",
                score=5,
            )
        ]
    )


def _coach_report() -> CoachReport:
    return CoachReport(
        summary="Strong, specific answers.", answer_advice=[], drills=[], study_plan="Keep going."
    )


def _report_state(
    *, grade: SessionGrade, coach: CoachReport | None, ended_early: bool = False
) -> WebState:
    return replace(
        interviewing("Tell me about a project."),
        question=None,
        finished=True,
        ended_early=ended_early,
        closing="Thanks for walking me through it.",
        grade=grade,
        coach=coach,
        transcript=[Turn(question="Tell me about a project.", answer="I led the migration.")],
    )


def test_a_full_session_resolves_to_coaching_with_no_banner() -> None:
    view = resolve_report_view(_report_state(grade=_scored_grade(), coach=_coach_report()))

    assert view.coaching is not None
    assert view.warning is None
    assert view.notice is None


def test_an_early_terminated_unscored_session_resolves_to_the_empty_notice() -> None:
    view = resolve_report_view(
        _report_state(grade=SessionGrade(scores=[]), coach=None, ended_early=True)
    )

    assert view.coaching is None
    assert view.warning is None
    assert view.notice == ENDED_EARLY_EMPTY_MESSAGE


def test_an_early_terminated_scored_session_warns_but_still_coaches() -> None:
    view = resolve_report_view(
        _report_state(grade=_scored_grade(), coach=_coach_report(), ended_early=True)
    )

    assert view.coaching is not None
    assert view.warning == ENDED_EARLY_PARTIAL_MESSAGE
    assert view.notice is None


def test_a_finished_session_without_coaching_resolves_to_the_no_coaching_notice() -> None:
    view = resolve_report_view(_report_state(grade=SessionGrade(scores=[]), coach=None))

    assert view.coaching is None
    assert view.warning is None
    assert view.notice == NO_COACHING_MESSAGE


def test_a_scored_session_whose_coaching_failed_warns_and_falls_back() -> None:
    view = resolve_report_view(_report_state(grade=_scored_grade(), coach=None, ended_early=True))

    assert view.coaching is None
    assert view.warning == ENDED_EARLY_PARTIAL_MESSAGE
    assert view.notice == NO_COACHING_MESSAGE


def test_an_early_terminated_unscored_session_shows_one_clear_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[str] = []
    monkeypatch.setitem(sys.modules, "streamlit", _fake_report_streamlit(shown))
    state = replace(
        interviewing("Tell me about a project."),
        ended_early=True,
        finished=True,
        question=None,
        transcript=[Turn(question="Just to confirm the scope?", answer="Yes, the whole platform.")],
        grade=SessionGrade(scores=[]),
        coach=None,
    )

    render_report_view(state)

    assert shown == [ENDED_EARLY_EMPTY_MESSAGE]


def test_a_full_report_renders_the_coaching_summary_and_no_banners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[str] = []
    markdowns: list[str] = []
    monkeypatch.setitem(sys.modules, "streamlit", _fake_report_streamlit(shown, markdowns))

    render_report_view(_report_state(grade=_scored_grade(), coach=_coach_report()))

    assert "Strong, specific answers." in markdowns
    assert shown == []


def test_chat_messages_pair_each_answered_turn_then_show_the_open_question() -> None:
    state = interviewing("Tell me about a project.")
    advanced = state_after_answer(state, "I led the migration.", TurnResult(next_question="Why?"))

    assert chat_messages(advanced) == [
        ("assistant", "Tell me about a project."),
        ("user", "I led the migration."),
        ("assistant", "Why?"),
    ]


def test_chat_messages_end_on_the_closing_line_when_finished() -> None:
    state = interviewing("Tell me about a project.")
    finished = state_after_answer(
        state,
        "I led the migration.",
        TurnResult(next_question=None, closing="Thanks for that.", transcript=[]),
    )

    assert chat_messages(finished) == [
        ("assistant", "Tell me about a project."),
        ("user", "I led the migration."),
        ("assistant", "Thanks for that."),
    ]


def _closing_beat(closing: str = "Thanks for walking me through it.") -> WebState:
    return state_after_answer(
        interviewing("Tell me about a project."),
        "I led the migration.",
        TurnResult(next_question=None, closing=closing, report_pending=True, transcript=[]),
    )


def test_a_closing_beat_is_its_own_phase_distinct_from_interview_and_report() -> None:
    state = _closing_beat()

    assert phase_of(state) == "closing"
    assert not state.finished


def test_the_closing_beat_renders_the_goodbye_as_the_final_assistant_bubble() -> None:
    state = _closing_beat()

    messages = chat_messages(state)

    assert messages == [
        ("assistant", "Tell me about a project."),
        ("user", "I led the migration."),
        ("assistant", "Thanks for walking me through it."),
    ]
    assert messages[-1] == ("assistant", "Thanks for walking me through it.")


def test_finalizing_a_closing_beat_carries_the_report_and_resolves_to_the_report_view() -> None:
    grade = _scored_grade()
    coach = _coach_report()
    transcript = [Turn(question="Tell me about a project.", answer="I led the migration.")]

    finished = state_after_finalize(
        _closing_beat(),
        TurnResult(
            next_question=None,
            closing="Thanks for walking me through it.",
            grade=grade,
            coach=coach,
            transcript=transcript,
        ),
    )

    assert phase_of(finished) == "report"
    assert finished.grade is grade
    assert finished.coach is coach
    assert finished.transcript == transcript
    view = resolve_report_view(finished)
    assert view.coaching is coach
    assert view.notice is None


class _FakeSessionState(dict[str, object]):
    """Stand-in for st.session_state: supports both attribute and item access, plus pop."""

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value


def _fake_interview_streamlit(
    session_state: _FakeSessionState, query_params: dict[str, str] | None = None
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        caption=lambda *a, **k: None,
        chat_message=lambda role: types.SimpleNamespace(write=lambda content: None),
        chat_input=lambda *a, **k: None,
        spinner=lambda *a, **k: _FakeStatusContext(),
        status=lambda *a, **k: _FakeStatusContext(),
        button=lambda *a, **k: True,
        error=lambda *a, **k: None,
        rerun=lambda *a, **k: None,
        query_params=query_params if query_params is not None else {},
        session_state=session_state,
    )


class _TwoBeatEngine:
    """Engine double whose every answer wraps to the closing beat, then finalizes to a report."""

    def __init__(self) -> None:
        self.finalized = False

    def submit_answer(self, thread_id: str, answer: str) -> TurnResult:
        return TurnResult(
            next_question=None,
            closing="That's a wrap, thanks.",
            report_pending=True,
            transcript=[Turn(question="Tell me about a project.", answer=answer)],
        )

    def finalize_report(self, thread_id: str) -> TurnResult:
        self.finalized = True
        return TurnResult(
            next_question=None,
            closing="That's a wrap, thanks.",
            grade=_scored_grade(),
            coach=_coach_report(),
            transcript=[Turn(question="Tell me about a project.", answer="I led the migration.")],
        )


def test_render_closing_finalizes_and_advances_to_the_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_state = _FakeSessionState()
    query_params: dict[str, str] = {}
    monkeypatch.setitem(
        sys.modules, "streamlit", _fake_interview_streamlit(session_state, query_params)
    )
    engine = _TwoBeatEngine()

    _render_closing(engine, _closing_beat())  # type: ignore[arg-type]

    assert engine.finalized
    advanced = session_state["state"]
    assert isinstance(advanced, WebState)
    assert phase_of(advanced) == "report"
    assert advanced.grade is not None and advanced.coach is not None
    assert "history_items" not in session_state
    assert query_params[SESSION_PARAM] == "t1"


def test_test_autopilot_drives_through_the_closing_beat_to_the_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOTELLME_TEST_MODE", "1")
    session_state = _FakeSessionState()
    monkeypatch.setitem(sys.modules, "streamlit", _fake_interview_streamlit(session_state))
    engine = _TwoBeatEngine()

    _render_test_autopilot(engine, interviewing("Tell me about a project."))  # type: ignore[arg-type]

    assert engine.finalized
    advanced = session_state["state"]
    assert isinstance(advanced, WebState)
    assert phase_of(advanced) == "report"
    assert advanced.grade is not None and advanced.coach is not None


class _FakeStatus:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []
        self.writes: list[str] = []

    def update(self, **kwargs: object) -> None:
        self.updates.append(kwargs)

    def write(self, line: str) -> None:
        self.writes.append(line)


def test_progress_keeps_the_status_expanded_on_each_new_stage() -> None:
    progress = _ModelProgress()
    status = _FakeStatus()
    progress.aim(status, "Reading your CV")

    progress.on_chat_model_start(tags=[f"{AGENT_TAG_PREFIX}parser"])
    progress.on_chat_model_start(tags=[f"{AGENT_TAG_PREFIX}grader"])

    assert all(update.get("expanded") is True for update in status.updates)
    assert len(status.updates) == 2


def test_progress_label_carries_no_call_count_suffix() -> None:
    progress = _ModelProgress()
    status = _FakeStatus()
    progress.aim(status, "Reading your CV")

    for _ in range(4):
        progress.on_chat_model_start(tags=[f"{AGENT_TAG_PREFIX}researcher"])

    labels = [update.get("label") for update in status.updates]
    assert labels == ["Reading your CV"] * 4


def test_progress_writes_one_line_per_distinct_phase_and_collapses_repeats() -> None:
    progress = _ModelProgress()
    status = _FakeStatus()
    progress.aim(status, "Reading your CV")

    progress.on_chat_model_start(tags=[f"{AGENT_TAG_PREFIX}researcher"])
    progress.on_chat_model_start(tags=[f"{AGENT_TAG_PREFIX}researcher"])
    progress.on_chat_model_start(tags=[f"{AGENT_TAG_PREFIX}director"])

    assert status.writes == ["Researching the company", "Choosing the next question"]


def test_agent_step_label_maps_a_tagged_agent_to_a_friendly_line() -> None:
    assert agent_step_label([f"{AGENT_TAG_PREFIX}grader"]) == "Grading your answers"
    assert agent_step_label([f"{AGENT_TAG_PREFIX}coach"]) == "Writing your coaching"


def test_agent_step_label_ignores_unrelated_or_missing_tags() -> None:
    assert agent_step_label(["some:other-tag"]) is None
    assert agent_step_label([f"{AGENT_TAG_PREFIX}unknown"]) is None
    assert agent_step_label(None) is None
    assert agent_step_label([]) is None


def test_blank_posting_text_becomes_no_posting() -> None:
    assert clean_posting("   \n  ") is None


def test_posting_text_is_trimmed() -> None:
    assert clean_posting("  We are hiring a backend engineer.  ") == (
        "We are hiring a backend engineer."
    )


def test_save_upload_writes_the_bytes_keeping_the_suffix(tmp_path: Path) -> None:
    path = save_upload("my-cv.pdf", b"%PDF-1.4 fake", tmp_path)

    assert path.parent == tmp_path
    assert path.suffix == ".pdf"
    assert path.read_bytes() == b"%PDF-1.4 fake"


def test_model_choices_lists_every_provider_model_pair() -> None:
    choices = model_choices(default_catalog())

    assert "anthropic:claude-opus-4-8" in choices
    assert "openai:gpt-5.5" in choices
    assert DEFAULT_CHOICE not in choices


def test_selections_become_agent_overrides_skipping_defaults() -> None:
    overrides = agent_overrides_from_selections(
        {"grader": "openai:gpt-5.5", "coach": DEFAULT_CHOICE}
    )

    assert overrides == {"grader": AgentModel(provider="openai", model="gpt-5.5")}


def test_default_provider_prefers_the_env_selection() -> None:
    assert default_provider(default_catalog(), {"SOTELLME_PROVIDER": "openai"}) == "openai"


def test_default_provider_falls_back_to_a_keyed_provider() -> None:
    assert default_provider(default_catalog(), {"OPENAI_API_KEY": "k"}) == "openai"


def test_default_provider_is_none_without_any_signal() -> None:
    assert default_provider(default_catalog(), {}) is None


def test_a_link_posting_is_marked_for_fetching() -> None:
    assert posting_to_resolve(LINK_MODE, "https://jobs.example.com/x", "") == (
        "https://jobs.example.com/x",
        True,
    )


def test_pasted_posting_text_is_used_as_is() -> None:
    assert posting_to_resolve(TEXT_MODE, "", "  We are hiring.  ") == ("We are hiring.", False)


def test_an_empty_posting_resolves_to_nothing() -> None:
    assert posting_to_resolve(LINK_MODE, "   ", "ignored") == (None, True)


class _StopRun(Exception):
    pass


def _fake_streamlit(shown: list[str]) -> types.SimpleNamespace:
    def stop() -> None:
        raise _StopRun()

    return types.SimpleNamespace(error=shown.append, warning=shown.append, stop=stop)


def test_tracing_callbacks_are_empty_when_langfuse_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit([]))
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert _tracing_callbacks() == []


def test_tracing_on_without_the_package_warns_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[str] = []
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit(shown))
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setattr(sotellme.tracing, "find_spec", lambda name: None)

    assert _tracing_callbacks() == []
    assert shown and "sotellme[web,tracing]" in shown[0]


class _OpenByIdEngine:
    """Engine double that reopens a thread by id, raising for an unknown one."""

    def __init__(self, snapshots: dict[str, SessionSnapshot]) -> None:
        self._snapshots = snapshots

    def snapshot(self, thread_id: str) -> SessionSnapshot:
        try:
            return self._snapshots[thread_id]
        except KeyError as exc:
            raise EngineError("This session didn't get far enough to reopen.") from exc


def _fake_recover_streamlit(
    session_state: _FakeSessionState,
    query_params: dict[str, str],
    warnings: list[str],
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        session_state=session_state,
        query_params=query_params,
        warning=warnings.append,
    )


def _finished_snapshot(thread_id: str) -> SessionSnapshot:
    return SessionSnapshot(
        thread_id=thread_id,
        question=None,
        needs_level=False,
        profile=profile(),
        finished=True,
        closing="Thanks.",
        grade=SessionGrade(scores=[]),
        coach=CoachReport(summary="s", answer_advice=[], drills=[], study_plan=""),
    )


def test_recover_reopens_a_finished_thread_named_in_the_url_to_its_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _OpenByIdEngine({"done": _finished_snapshot("done")})
    monkeypatch.setattr(sotellme.web, "_recovery_engine", lambda catalog: engine)
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        _fake_recover_streamlit(_FakeSessionState(), {SESSION_PARAM: "done"}, []),
    )

    state = _recover(default_catalog())

    assert state is not None
    assert phase_of(state) == "report"
    assert state.thread_id == "done"


def test_recover_reopens_an_in_progress_thread_named_in_the_url_to_its_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = SessionSnapshot(
        thread_id="live",
        question="Tell me about a project.",
        needs_level=False,
        profile=profile(),
    )
    engine = _OpenByIdEngine({"live": snapshot})
    monkeypatch.setattr(sotellme.web, "_recovery_engine", lambda catalog: engine)
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        _fake_recover_streamlit(_FakeSessionState(), {SESSION_PARAM: "live"}, []),
    )

    state = _recover(default_catalog())

    assert state is not None
    assert phase_of(state) == "interview"
    assert state.question == "Tell me about a project."


def test_recover_without_a_session_param_is_a_cold_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A bare visit starts fresh, never auto-resuming the latest session.
    engine = _OpenByIdEngine({"live": _finished_snapshot("live")})
    monkeypatch.setattr(sotellme.web, "_recovery_engine", lambda catalog: engine)
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        _fake_recover_streamlit(_FakeSessionState(), {}, []),
    )

    assert _recover(default_catalog()) is None


def test_recover_clears_a_blank_session_param_on_cold_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_params = {SESSION_PARAM: ""}
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        _fake_recover_streamlit(_FakeSessionState(), query_params, []),
    )

    assert _recover(default_catalog()) is None
    assert SESSION_PARAM not in query_params


def test_recover_degrades_to_setup_when_the_url_thread_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _OpenByIdEngine({})
    monkeypatch.setattr(sotellme.web, "_recovery_engine", lambda catalog: engine)
    warnings: list[str] = []
    query_params = {SESSION_PARAM: "ghost"}
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        _fake_recover_streamlit(_FakeSessionState(), query_params, warnings),
    )

    assert _recover(default_catalog()) is None
    assert warnings
    assert SESSION_PARAM not in query_params


def test_opening_a_session_anchors_it_in_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _OpenByIdEngine({"past": _finished_snapshot("past")})
    monkeypatch.setattr(sotellme.web, "_recovery_engine", lambda catalog: engine)
    query_params: dict[str, str] = {}
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        _fake_recover_streamlit(_FakeSessionState(), query_params, []),
    )

    state = _open_session(default_catalog(), "past")

    assert state is not None
    assert query_params[SESSION_PARAM] == "past"


def test_starting_a_session_anchors_it_before_the_rerun(monkeypatch: pytest.MonkeyPatch) -> None:
    # The anchor must land in the URL at start, so a reconnect that drops the
    # in-memory session can still recover the thread from the URL.
    session_state = _FakeSessionState()
    session_state["new_interview"] = True
    query_params: dict[str, str] = {}
    monkeypatch.setitem(
        sys.modules, "streamlit", _fake_new_interview_streamlit(session_state, query_params)
    )
    snapshot = SessionSnapshot(
        thread_id="fresh",
        question="Tell me about a project.",
        needs_level=False,
        profile=profile(),
    )

    _begin_session(_OpenByIdEngine({}), snapshot)  # type: ignore[arg-type]

    assert query_params[SESSION_PARAM] == "fresh"
    assert isinstance(session_state["state"], WebState)
    assert "engine" in session_state
    assert "new_interview" not in session_state


def test_anchor_session_writes_the_thread_id_to_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    query_params: dict[str, str] = {}
    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace(query_params=query_params))

    _anchor_session("t42")

    assert query_params[SESSION_PARAM] == "t42"


def _fake_new_interview_streamlit(
    session_state: _FakeSessionState, query_params: dict[str, str]
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        session_state=session_state,
        query_params=query_params,
        rerun=lambda *a, **k: None,
    )


def test_starting_a_new_interview_clears_the_url_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    session_state = _FakeSessionState()
    session_state["state"] = interviewing("Tell me about a project.")
    query_params = {SESSION_PARAM: "t1"}
    monkeypatch.setitem(
        sys.modules, "streamlit", _fake_new_interview_streamlit(session_state, query_params)
    )

    _start_new_interview()

    assert SESSION_PARAM not in query_params
    assert "state" not in session_state
    assert session_state["new_interview"] is True

import sys
import types
from pathlib import Path

import pytest

import sotellme.tracing
from sotellme.catalog import default_catalog
from sotellme.coach import CoachReport
from sotellme.config import AGENT_TAG_PREFIX, AgentModel
from sotellme.engine import SessionSnapshot, TurnResult
from sotellme.grader import SessionGrade
from sotellme.interviewer import Turn
from sotellme.profile import CandidateProfile, Role
from sotellme.web import (
    DEFAULT_CHOICE,
    LINK_MODE,
    TEXT_MODE,
    WebState,
    _tracing_callbacks,
    agent_overrides_from_selections,
    agent_step_label,
    chat_messages,
    clean_posting,
    default_provider,
    model_choices,
    phase_of,
    posting_to_resolve,
    save_upload,
    state_after_answer,
    state_from_snapshot,
)


def profile() -> CandidateProfile:
    return CandidateProfile(
        roles=[Role(title="Engineer", organization="Acme")],
        projects=[],
        quantified_claims=[],
        technologies=[],
    )


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

    return types.SimpleNamespace(error=shown.append, stop=stop)


def test_tracing_callbacks_are_empty_when_langfuse_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit([]))
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert _tracing_callbacks() == []


def test_tracing_on_without_the_package_shows_a_message_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: list[str] = []
    monkeypatch.setitem(sys.modules, "streamlit", _fake_streamlit(shown))
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setattr(sotellme.tracing, "find_spec", lambda name: None)

    with pytest.raises(_StopRun):
        _tracing_callbacks()

    assert shown and "sotellme[tracing]" in shown[0]

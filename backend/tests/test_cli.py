from pathlib import Path

import pytest
from rich.console import Console

import sotellme.cli as cli
import sotellme.tracing
from sotellme.assessor import StarFlags
from sotellme.cli import (
    NO_REPORTS_MESSAGE,
    TranscriptInputError,
    ask_target_level,
    build_engine,
    build_parser,
    format_cost_estimate,
    format_report_list,
    format_score_summary,
    parse_target_level,
    parse_transcript,
    read_multiline_answer,
    streamlit_run_command,
    strip_done_sentinel,
)
from sotellme.coach import CoachReport
from sotellme.config import AGENT_ROLES, resolve_model_config
from sotellme.engine import SessionSnapshot, TurnResult
from sotellme.grader import AnswerScore, SessionGrade
from sotellme.guardrail import GuardrailError
from sotellme.interviewer import Turn
from sotellme.pricing import CostEstimate, CostSummary, ModelCost, format_cost_summary
from sotellme.profile import CandidateProfile, Role


def score(
    question: str,
    *,
    weak_or_missing: list[str] | None = None,
    gap: str = "",
    ownership: str = "clear",
    value: int = 5,
) -> AnswerScore:
    resolved_gap = "" if value == 5 else (gap or "One refinement short of a five.")
    return AnswerScore(
        question=question,
        star=StarFlags(situation=True, task=True, action=True, result=True, quantified_result=True),
        specificity="high",
        ownership=ownership,  # type: ignore[arg-type]
        weak_or_missing=weak_or_missing or [],  # type: ignore[arg-type]
        gap=resolved_gap,
        rationale="Complete, quantified story at the target level.",
        score=value,
    )


def test_strip_done_sentinel_removes_a_trailing_done_line() -> None:
    assert strip_done_sentinel("My answer.\n/done") == "My answer."


def test_strip_done_sentinel_leaves_plain_text_untouched() -> None:
    assert strip_done_sentinel("My answer.") == "My answer."


def test_strip_done_sentinel_preserves_internal_blank_lines() -> None:
    assert strip_done_sentinel("Paragraph one.\n\nParagraph two.\n/done") == (
        "Paragraph one.\n\nParagraph two."
    )


def test_strip_done_sentinel_handles_trailing_whitespace_around_done() -> None:
    assert strip_done_sentinel("My answer.\n /done \n\n") == "My answer."


def test_answer_ends_at_blank_line() -> None:
    lines = iter(["First line.", "Second line.", ""])

    answer = read_multiline_answer(lambda: next(lines))

    assert answer == "First line.\nSecond line."


def test_answer_ends_at_done_marker() -> None:
    lines = iter(["Only line.", "/done", "never read"])

    answer = read_multiline_answer(lambda: next(lines))

    assert answer == "Only line."


def test_answer_ends_at_eof() -> None:
    lines = iter(["Only line."])

    def read_line() -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    assert read_multiline_answer(read_line) == "Only line."


def test_interview_command_parses_cv_and_model_flags() -> None:
    args = build_parser().parse_args(
        ["interview", "--cv", "cv.pdf", "--provider", "anthropic", "--fast-model", "m"]
    )

    assert args.command == "interview"
    assert args.cv == "cv.pdf"
    assert args.provider == "anthropic"
    assert args.fast_model == "m"
    assert args.smart_model is None
    assert args.job is None


def test_interview_command_parses_the_job_posting() -> None:
    args = build_parser().parse_args(["interview", "--cv", "cv.pdf", "--job", "posting.txt"])

    assert args.job == "posting.txt"


def test_target_levels_parse_case_insensitively() -> None:
    assert parse_target_level("Senior") == "senior"
    assert parse_target_level("  MID ") == "mid"
    assert parse_target_level("junior") == "junior"
    assert parse_target_level("staff") == "staff"


def test_anything_else_is_not_a_level() -> None:
    assert parse_target_level("principal") is None
    assert parse_target_level("") is None


def test_the_level_is_asked_until_an_answer_is_valid() -> None:
    lines = iter(["principal", "", "senior"])

    assert ask_target_level(lambda: next(lines), lambda message: None) == "senior"


def test_resume_command_parses() -> None:
    args = build_parser().parse_args(["resume"])

    assert args.command == "resume"


def test_reports_command_parses() -> None:
    args = build_parser().parse_args(["reports"])

    assert args.command == "reports"


def test_web_command_parses() -> None:
    args = build_parser().parse_args(["web"])

    assert args.command == "web"


def test_streamlit_run_command_launches_the_web_module_with_a_calm_accent() -> None:
    command = streamlit_run_command(Path("/pkg/web.py"), "/usr/bin/python")

    assert command[:5] == ["/usr/bin/python", "-m", "streamlit", "run", "/pkg/web.py"]
    assert command[5] == "--theme.primaryColor=#4f6d7a"
    assert command[6] == "--client.toolbarMode=minimal"


def test_build_engine_wires_every_agent_from_its_own_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SOTELLME_DATA_DIR", str(tmp_path))
    requested: list[str] = []
    monkeypatch.setattr(cli, "build_chat_model", lambda config, key: requested.append(key))
    config = resolve_model_config(provider="anthropic", env={"ANTHROPIC_API_KEY": "k"})

    build_engine(config, [])

    assert set(requested) == set(AGENT_ROLES)


def test_tracing_on_without_the_package_warns_and_continues(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setattr(sotellme.tracing, "find_spec", lambda name: None)

    assert cli._tracing_callbacks() == []
    assert "sotellme[web,tracing]" in capsys.readouterr().err


def test_format_report_list_names_each_report() -> None:
    listing = format_report_list(
        [
            Path("sotellme-report-20260614-080000.md"),
            Path("sotellme-report-20260610-080000.md"),
        ]
    )

    assert "sotellme-report-20260614-080000.md" in listing
    assert "sotellme-report-20260610-080000.md" in listing


def test_format_report_list_handles_no_reports() -> None:
    assert format_report_list([]) == NO_REPORTS_MESSAGE


def test_grade_command_parses_transcript_level_and_model_flags() -> None:
    args = build_parser().parse_args(
        ["grade", "session.json", "--level", "senior", "--provider", "anthropic"]
    )

    assert args.command == "grade"
    assert args.transcript == "session.json"
    assert args.level == "senior"
    assert args.provider == "anthropic"


def test_parse_transcript_reads_question_answer_pairs_in_order() -> None:
    turns = parse_transcript(
        '[{"question": "Tell me about a project.", "answer": "I led the migration."},'
        ' {"question": "What was hard?", "answer": "The data parity."}]'
    )

    assert [turn.question for turn in turns] == ["Tell me about a project.", "What was hard?"]
    assert turns[0].answer == "I led the migration."


def test_parse_transcript_rejects_malformed_json() -> None:
    with pytest.raises(TranscriptInputError):
        parse_transcript("not json at all")


def test_parse_transcript_rejects_a_non_list_document() -> None:
    with pytest.raises(TranscriptInputError):
        parse_transcript('{"question": "q", "answer": "a"}')


def test_parse_transcript_rejects_a_turn_missing_a_field() -> None:
    with pytest.raises(TranscriptInputError):
        parse_transcript('[{"question": "q"}]')


def test_parse_transcript_rejects_an_empty_transcript() -> None:
    with pytest.raises(TranscriptInputError):
        parse_transcript("[]")


def test_score_summary_lists_each_answer_with_its_score_and_named_gaps() -> None:
    grade = SessionGrade(
        scores=[
            score(
                "Tell me about the migration.",
                weak_or_missing=["result"],
                gap="No outcome stated.",
                value=3,
            ),
            score("What problem was it solving?", value=5),
        ]
    )

    summary = format_score_summary(grade)

    assert "Tell me about the migration." in summary
    assert "3/5" in summary
    assert "5/5" in summary
    assert "result" in summary
    assert "No outcome stated." in summary


def test_score_summary_omits_ownership_when_not_applicable() -> None:
    grade = SessionGrade(
        scores=[score("Why do you want to work here?", ownership="not_applicable", value=4)]
    )

    summary = format_score_summary(grade)

    assert "specificity" in summary
    assert "ownership" not in summary


def test_score_summary_handles_a_session_with_no_scored_answers() -> None:
    summary = format_score_summary(SessionGrade(scores=[]))

    assert summary.strip()


def test_cost_estimate_names_the_dollar_amount_model_and_that_it_is_an_estimate() -> None:
    estimate = CostEstimate(
        model="claude-opus-4-8",
        expected_turns=12,
        input_tokens=62_000,
        output_tokens=13_900,
        usd=0.66,
    )

    line = format_cost_estimate(estimate)

    assert "$0.66" in line
    assert "claude-opus-4-8" in line
    assert "estimate" in line.lower()


def test_cost_estimate_is_honest_when_a_model_has_no_price() -> None:
    estimate = CostEstimate(
        model="mystery",
        expected_turns=12,
        input_tokens=1,
        output_tokens=1,
        usd=None,
    )

    line = format_cost_estimate(estimate)

    assert "mystery" in line
    assert "$" not in line


def test_cost_summary_reports_tokens_and_an_estimated_dollar_cost() -> None:
    summary = format_cost_summary(
        CostSummary(
            per_model=(ModelCost("claude-opus-4-8", 70_000, 5_900, 0.66),),
            total_tokens=75_900,
            usd=0.66,
        )
    )

    assert "75,900" in summary
    assert "$0.66" in summary
    assert "estimate" in summary.lower()


def test_cost_summary_breaks_each_model_into_input_and_output() -> None:
    summary = format_cost_summary(
        CostSummary(
            per_model=(
                ModelCost("claude-sonnet-4-6", 120_000, 9_000, 0.49),
                ModelCost("claude-opus-4-8", 8_000, 4_000, 0.14),
            ),
            total_tokens=141_000,
            usd=0.63,
        )
    )

    assert "claude-sonnet-4-6" in summary
    assert "claude-opus-4-8" in summary
    assert "120,000" in summary
    assert "9,000" in summary
    assert "$0.49" in summary


def test_cost_summary_reports_cached_tokens_and_estimated_savings() -> None:
    summary = format_cost_summary(
        CostSummary(
            per_model=(
                ModelCost("claude-opus-4-8", 70_000, 5_900, 0.40, cached_input_tokens=50_000),
            ),
            total_tokens=75_900,
            usd=0.40,
            saved_usd=0.30,
        )
    )

    assert "50,000 cached" in summary
    assert "$0.30" in summary
    assert "saved" in summary.lower()


def test_cost_summary_stays_quiet_about_caching_when_nothing_was_cached() -> None:
    summary = format_cost_summary(
        CostSummary(
            per_model=(ModelCost("claude-opus-4-8", 70_000, 5_900, 0.66),),
            total_tokens=75_900,
            usd=0.66,
        )
    )

    assert "cached" not in summary.lower()
    assert "saved" not in summary.lower()


class _FailingStartEngine:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def __enter__(self) -> "_FailingStartEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def start(self, *args: object, **kwargs: object) -> object:
        raise self._error


def test_main_reports_a_guardrail_failure_as_a_clean_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SOTELLME_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SOTELLME_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    failing = _FailingStartEngine(GuardrailError("Could not screen."))
    monkeypatch.setattr(cli, "build_engine", lambda *a, **k: failing)

    code = cli.main(["interview", "--cv", "cv.md"])

    assert code == 1
    assert "Could not screen." in capsys.readouterr().out


def _profile() -> CandidateProfile:
    return CandidateProfile(
        roles=[Role(title="Engineer", organization="Acme")],
        projects=[],
        quantified_claims=[],
        technologies=[],
    )


class _FinishingEngine:
    def __init__(self, grade: SessionGrade, coach: CoachReport) -> None:
        self._grade = grade
        self._coach = coach

    def submit_answer(self, thread_id: str, answer: str) -> TurnResult:
        return TurnResult(
            next_question=None,
            closing="Thanks for walking me through it.",
            grade=self._grade,
            coach=self._coach,
            transcript=[Turn(question="Tell me about a project.", answer=answer)],
        )

    def session_usage(self) -> dict[str, object]:
        return {}


def test_run_session_survives_an_unwritable_report_directory(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *a: "")

    def unwritable(*args: object, **kwargs: object) -> Path:
        raise OSError("Read-only file system")

    monkeypatch.setattr(cli, "write_report", unwritable)
    grade = SessionGrade(scores=[score("Tell me about a project.")])
    coach = CoachReport(summary="Tighten the result.", answer_advice=[], drills=[], study_plan="")
    engine = _FinishingEngine(grade, coach)
    session = SessionSnapshot(
        thread_id="t1",
        profile=_profile(),
        needs_level=False,
        question="Tell me about a project.",
    )

    cli._run_session(Console(), engine, session, {})  # type: ignore[arg-type]

    assert "Read-only file system" in capsys.readouterr().out


def test_cost_summary_flags_models_it_could_not_price() -> None:
    summary = format_cost_summary(
        CostSummary(
            per_model=(ModelCost("mystery", 500, 500, None),),
            total_tokens=1_000,
            usd=0.0,
        )
    )

    assert "mystery" in summary
    assert "price not configured" in summary

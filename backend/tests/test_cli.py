from pathlib import Path

import pytest

import sotellme.cli as cli
from sotellme.assessor import StarFlags
from sotellme.cli import (
    NO_REPORTS_MESSAGE,
    TranscriptInputError,
    ask_target_level,
    build_engine,
    build_parser,
    format_report_list,
    format_score_summary,
    parse_target_level,
    parse_transcript,
    read_multiline_answer,
    streamlit_run_command,
    strip_done_sentinel,
)
from sotellme.config import AGENT_ROLES, resolve_model_config
from sotellme.grader import AnswerScore, SessionGrade


def score(
    question: str,
    *,
    weak_or_missing: list[str] | None = None,
    gap: str = "",
    ownership: str = "clear",
    value: int = 5,
) -> AnswerScore:
    return AnswerScore(
        question=question,
        star=StarFlags(situation=True, task=True, action=True, result=True, quantified_result=True),
        specificity="high",
        ownership=ownership,  # type: ignore[arg-type]
        weak_or_missing=weak_or_missing or [],  # type: ignore[arg-type]
        gap=gap,
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

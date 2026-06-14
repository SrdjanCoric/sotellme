from sotellme.assessor import StarFlags
from sotellme.cli import (
    ask_target_level,
    build_parser,
    format_score_summary,
    parse_target_level,
    read_multiline_answer,
    strip_done_sentinel,
)
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

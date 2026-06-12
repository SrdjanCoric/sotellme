from sotellme.cli import ask_target_level, build_parser, parse_target_level, read_multiline_answer


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

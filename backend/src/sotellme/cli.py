import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from rich.console import Console
from rich.panel import Panel

from sotellme.assessor import AssessorError, assess_answer
from sotellme.catalog import CatalogError, load_catalog
from sotellme.coach import CoachingError, CoachReport, coach_session
from sotellme.config import ModelConfig, ModelConfigError, build_chat_model, resolve_model_config
from sotellme.director import DirectorError, LLMDirector
from sotellme.engine import (
    Assessor,
    Coacher,
    Director,
    EngineError,
    Grader,
    InterviewEngine,
    Interviewer,
    ProfileParser,
    Researcher,
    RoleBuilder,
    SessionSnapshot,
)
from sotellme.extraction import CVInputError
from sotellme.fetch import fetch_research_page
from sotellme.grader import GradingError, SessionGrade, grade_session
from sotellme.interviewer import LLMInterviewer, Turn
from sotellme.posting import PostingInputError, resolve_posting_text
from sotellme.profile import ProfileParseError, parse_candidate_profile
from sotellme.report import list_reports, write_report
from sotellme.research import build_company_brief
from sotellme.role import RoleContext, RoleContextError, TargetLevel, build_role_context
from sotellme.tracing import TracingError, langfuse_callbacks

NO_SCORES_MESSAGE = "No answers to score from this session."

NO_COACHING_MESSAGE = "No coaching for this session; there were no answers to work from."

NO_REPORTS_MESSAGE = "No coaching reports in this directory yet."

_STAR_LABELS = {
    "situation": "situation",
    "task": "task",
    "action": "action",
    "result": "result",
    "quantified_result": "a number on the result",
}


def _truncate_question(question: str, limit: int = 80) -> str:
    collapsed = " ".join(question.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def format_score_summary(grade: SessionGrade) -> str:
    if not grade.scores:
        return NO_SCORES_MESSAGE
    blocks: list[str] = []
    for index, answer in enumerate(grade.scores, start=1):
        lines = [f"{index}. [{answer.score}/5] {_truncate_question(answer.question)}"]
        credibility = f"   specificity: {answer.specificity}"
        if answer.ownership != "not_applicable":
            credibility += f" · ownership: {answer.ownership}"
        lines.append(credibility)
        if answer.weak_or_missing:
            named = ", ".join(_STAR_LABELS[element] for element in answer.weak_or_missing)
            lines.append(f"   weak or missing: {named}")
        if answer.gap:
            lines.append(f"   {answer.gap}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_report_list(reports: Sequence[Path]) -> str:
    if not reports:
        return NO_REPORTS_MESSAGE
    return "\n".join(report.name for report in reports)


LEVEL_PROMPT = "What level is this interview for? (junior / mid / senior / staff)"

TARGET_LEVELS: tuple[TargetLevel, ...] = ("junior", "mid", "senior", "staff")

PLAIN_ANSWER_HINT = "[dim]Answer below. End with a blank line or /done.[/dim]"

RICH_ANSWER_HINT = (
    "[dim]Enter for a new line · Esc then Enter to send · /done on its own line also sends.[/dim]"
)


def parse_target_level(raw: str) -> TargetLevel | None:
    cleaned = raw.strip().lower()
    if cleaned in TARGET_LEVELS:
        return cleaned
    return None


def ask_target_level(read_line: Callable[[], str], show: Callable[[str], None]) -> TargetLevel:
    show(LEVEL_PROMPT)
    while True:
        level = parse_target_level(read_line())
        if level is not None:
            return level
        show("Please answer junior, mid, senior, or staff.")


DONE_SENTINEL = "/done"


def strip_done_sentinel(text: str) -> str:
    lines = text.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip() == DONE_SENTINEL:
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


class TranscriptInputError(Exception):
    pass


def parse_transcript(text: str) -> list[Turn]:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TranscriptInputError(
            "The transcript file is not valid JSON. Expected a list of "
            '{"question": ..., "answer": ...} objects.'
        ) from exc
    if not isinstance(document, list) or not document:
        raise TranscriptInputError(
            "The transcript must be a non-empty JSON list of "
            '{"question": ..., "answer": ...} objects.'
        )
    turns: list[Turn] = []
    for index, entry in enumerate(document, start=1):
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("question"), str)
            or not isinstance(entry.get("answer"), str)
        ):
            raise TranscriptInputError(
                f"Turn {index} must have a string 'question' and a string 'answer'."
            )
        turns.append(Turn(question=entry["question"], answer=entry["answer"]))
    return turns


def read_multiline_answer(read_line: Callable[[], str]) -> str:
    lines: list[str] = []
    while True:
        try:
            line = read_line()
        except EOFError:
            break
        if not line.strip() or line.strip() == DONE_SENTINEL:
            break
        lines.append(line)
    return "\n".join(lines)


def _interactive() -> bool:
    return sys.stdin.isatty()


def _build_answer_session() -> PromptSession[str]:
    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit_or_newline(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        if buffer.document.current_line.strip() == DONE_SENTINEL:
            buffer.validate_and_handle()
        else:
            buffer.insert_text("\n")

    @bindings.add("escape", "enter")
    def _submit(event: KeyPressEvent) -> None:
        event.current_buffer.validate_and_handle()

    return PromptSession(multiline=True, key_bindings=bindings)


def _read_interactive_answer(session: PromptSession[str]) -> str:
    try:
        return strip_done_sentinel(session.prompt())
    except EOFError:
        return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sotellme", description="Behavioral interview simulator and coach."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    interview = subcommands.add_parser("interview", help="Start a new interview session.")
    interview.add_argument("--cv", required=True, help="Path to your CV (PDF, markdown, or text).")
    interview.add_argument(
        "--job",
        help="The job posting: a link, a file path, or the pasted posting text.",
    )
    _add_model_flags(interview)

    resume = subcommands.add_parser("resume", help="Resume the latest interrupted session.")
    _add_model_flags(resume)

    subcommands.add_parser("reports", help="List the coaching reports in this directory.")

    subcommands.add_parser("web", help="Launch the local web UI in your browser.")

    grade = subcommands.add_parser(
        "grade", help="Grade a saved transcript without running a live interview."
    )
    grade.add_argument(
        "transcript",
        help='Path to a JSON file: a list of {"question": ..., "answer": ...} objects.',
    )
    grade.add_argument(
        "--level",
        required=True,
        help="Target level to grade against: junior, mid, senior, or staff.",
    )
    _add_model_flags(grade)

    return parser


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", help="LLM provider: anthropic, openai, or google_genai.")
    parser.add_argument("--fast-model", help="Override the fast model slot.")
    parser.add_argument("--smart-model", help="Override the smart model slot.")


def _data_dir() -> Path:
    return Path(os.environ.get("SOTELLME_DATA_DIR", "~/.sotellme")).expanduser()


def _build_profile_parser(config: ModelConfig) -> ProfileParser:
    model = build_chat_model(config, "parser")
    return lambda cv_text: parse_candidate_profile(cv_text, model)


def _build_assessor(config: ModelConfig) -> Assessor:
    model = build_chat_model(config, "assessor")
    return lambda topic, transcript: assess_answer(topic, transcript, model)


def _build_director(config: ModelConfig) -> Director:
    return LLMDirector(build_chat_model(config, "director"))


def _build_interviewer(config: ModelConfig) -> Interviewer:
    return LLMInterviewer(build_chat_model(config, "interviewer"))


def _build_role_builder(config: ModelConfig) -> RoleBuilder:
    model = build_chat_model(config, "role_builder")
    return lambda posting_text: build_role_context(posting_text, model)


def _build_researcher(config: ModelConfig) -> Researcher:
    model = build_chat_model(config, "researcher")

    def research(posting_text: str, context: RoleContext) -> str:
        return build_company_brief(posting_text, context, model, fetch_research_page)

    return research


def _build_grader(config: ModelConfig) -> Grader:
    model = build_chat_model(config, "grader")
    return lambda transcript, target_level: grade_session(transcript, target_level, model)


def _build_coacher(config: ModelConfig) -> Coacher:
    model = build_chat_model(config, "coach")
    return lambda transcript, grade, target_level: coach_session(
        transcript, grade, target_level, model
    )


def build_engine(
    config: ModelConfig, callbacks: list[BaseCallbackHandler]
) -> InterviewEngine:
    return InterviewEngine(
        data_dir=_data_dir(),
        profile_parser=_build_profile_parser(config),
        assessor=_build_assessor(config),
        director=_build_director(config),
        interviewer=_build_interviewer(config),
        role_builder=_build_role_builder(config),
        researcher=_build_researcher(config),
        grader=_build_grader(config),
        coacher=_build_coacher(config),
        callbacks=callbacks,
    )


WEB_EXTRA_HINT = (
    "The web UI needs Streamlit, which isn't installed. Install the web extra:\n"
    "    uv sync --extra web\n"
    "or:\n"
    "    pip install 'sotellme[web]'"
)


WEB_ACCENT_COLOR = "#4f6d7a"


def streamlit_run_command(web_module: Path, executable: str) -> list[str]:
    return [
        executable,
        "-m",
        "streamlit",
        "run",
        str(web_module),
        f"--theme.primaryColor={WEB_ACCENT_COLOR}",
        "--client.toolbarMode=minimal",
    ]


def _run_web(console: Console) -> int:
    if importlib.util.find_spec("streamlit") is None:
        console.print(WEB_EXTRA_HINT, style="red", markup=False)
        return 1
    web_module = Path(__file__).with_name("web.py")
    return subprocess.call(streamlit_run_command(web_module, sys.executable))


def _run_session(console: Console, engine: InterviewEngine, session: SessionSnapshot) -> None:
    interactive = _interactive()
    answer_session = _build_answer_session() if interactive else None
    level_session: PromptSession[str] | None = PromptSession() if interactive else None

    if session.needs_level:
        read_level = (lambda: level_session.prompt("> ")) if level_session else input
        level = ask_target_level(read_level, console.print)
        with console.status("Starting the session..."):
            session = engine.submit_level(session.thread_id, level)
    question: str | None = session.question
    closing: str | None = None
    grade: SessionGrade | None = None
    coach: CoachReport | None = None
    transcript: list[Turn] = []
    while question is not None:
        console.print(Panel(question, title="Interviewer", border_style="cyan"))
        if answer_session is not None:
            console.print(RICH_ANSWER_HINT)
            answer = _read_interactive_answer(answer_session)
        else:
            console.print(PLAIN_ANSWER_HINT)
            answer = read_multiline_answer(input)
        with console.status("Thinking..."):
            result = engine.submit_answer(session.thread_id, answer)
        question = result.next_question
        closing = result.closing
        grade = result.grade
        coach = result.coach
        transcript = result.transcript
    if closing:
        console.print(Panel(closing, title="Interviewer", border_style="cyan"))
    if grade is not None:
        console.print(Panel(format_score_summary(grade), title="Scorecard", border_style="magenta"))
    if coach is not None and grade is not None and grade.scores:
        path = write_report(coach, transcript, Path.cwd(), datetime.now())
        if coach.summary.strip():
            console.print(Panel(coach.summary.strip(), title="Coaching", border_style="green"))
        console.print(f"\n[bold]Your full coaching report:[/bold] {path}")
    else:
        console.print(f"\n[dim]{NO_COACHING_MESSAGE}[/dim]")


def _run_grade(console: Console, config: ModelConfig, transcript_path: str, raw_level: str) -> None:
    level = parse_target_level(raw_level)
    if level is None:
        raise TranscriptInputError(
            f"'{raw_level}' is not a level. Use junior, mid, senior, or staff."
        )
    try:
        text = Path(transcript_path).read_text()
    except OSError as exc:
        raise TranscriptInputError(f"Could not read the transcript file: {exc}") from exc
    transcript = parse_transcript(text)
    model = build_chat_model(config, "grader")
    with console.status("Grading the transcript..."):
        grade = grade_session(transcript, level, model)
    console.print(Panel(format_score_summary(grade), title="Scorecard", border_style="magenta"))


def _run_reports(console: Console) -> None:
    console.print(format_report_list(list_reports(Path.cwd())))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    if args.command == "reports":
        _run_reports(console)
        return 0
    if args.command == "web":
        return _run_web(console)
    try:
        config = resolve_model_config(
            env=os.environ,
            provider=args.provider,
            fast_model=args.fast_model,
            smart_model=args.smart_model,
            catalog=load_catalog(_data_dir()),
        )
        if args.command == "grade":
            _run_grade(console, config, args.transcript, args.level)
            return 0
        posting_text = None
        if args.command == "interview" and args.job:
            posting_text = resolve_posting_text(args.job)
        callbacks = langfuse_callbacks(os.environ)
        engine = build_engine(config, callbacks)
        with engine:
            if args.command == "interview":
                status = (
                    "Reading your CV and researching the company..."
                    if posting_text
                    else "Reading your CV..."
                )
                with console.status(status):
                    session = engine.start(Path(args.cv), posting_text=posting_text)
            else:
                session = engine.resume_latest()
            _run_session(console, engine, session)
    except (
        ModelConfigError,
        CatalogError,
        CVInputError,
        PostingInputError,
        ProfileParseError,
        RoleContextError,
        AssessorError,
        DirectorError,
        GradingError,
        CoachingError,
        TranscriptInputError,
        EngineError,
        TracingError,
    ) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

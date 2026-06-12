import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from sotellme.assessor import AssessorError, assess_answer
from sotellme.config import ModelConfig, ModelConfigError, build_chat_model, resolve_model_config
from sotellme.director import DirectorError, LLMDirector
from sotellme.engine import (
    Assessor,
    Director,
    EngineError,
    InterviewEngine,
    Interviewer,
    ProfileParser,
    Researcher,
    RoleBuilder,
    SessionHandle,
)
from sotellme.extraction import CVInputError
from sotellme.fetch import fetch_research_page
from sotellme.interviewer import LLMInterviewer
from sotellme.posting import PostingInputError, resolve_posting_text
from sotellme.profile import ProfileParseError, parse_candidate_profile
from sotellme.research import build_company_brief
from sotellme.role import RoleContext, RoleContextError, TargetLevel, build_role_context
from sotellme.tracing import TracingError, langfuse_callbacks

CLOSING_MESSAGE = "That's a wrap. Grading and coaching arrive in a coming release."

LEVEL_PROMPT = "What level is this interview for? (junior / mid / senior / staff)"

TARGET_LEVELS: tuple[TargetLevel, ...] = ("junior", "mid", "senior", "staff")


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


def read_multiline_answer(read_line: Callable[[], str]) -> str:
    lines: list[str] = []
    while True:
        try:
            line = read_line()
        except EOFError:
            break
        if not line.strip() or line.strip() == "/done":
            break
        lines.append(line)
    return "\n".join(lines)


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

    return parser


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", help="LLM provider: anthropic, openai, or google_genai.")
    parser.add_argument("--fast-model", help="Override the fast model slot.")
    parser.add_argument("--smart-model", help="Override the smart model slot.")


def _data_dir() -> Path:
    return Path(os.environ.get("SOTELLME_DATA_DIR", "~/.sotellme")).expanduser()


def _build_profile_parser(config: ModelConfig) -> ProfileParser:
    model = build_chat_model(config, "fast")
    return lambda cv_text: parse_candidate_profile(cv_text, model)


def _build_assessor(config: ModelConfig) -> Assessor:
    model = build_chat_model(config, "fast")
    return lambda topic, transcript: assess_answer(topic, transcript, model)


def _build_director(config: ModelConfig) -> Director:
    return LLMDirector(build_chat_model(config, "fast"))


def _build_interviewer(config: ModelConfig) -> Interviewer:
    return LLMInterviewer(build_chat_model(config, "fast"))


def _build_role_builder(config: ModelConfig) -> RoleBuilder:
    model = build_chat_model(config, "fast")
    return lambda posting_text: build_role_context(posting_text, model)


def _build_researcher(config: ModelConfig) -> Researcher:
    model = build_chat_model(config, "fast")

    def research(posting_text: str, context: RoleContext) -> str:
        return build_company_brief(posting_text, context, model, fetch_research_page)

    return research


def _run_session(console: Console, engine: InterviewEngine, session: SessionHandle) -> None:
    if session.needs_level:
        level = ask_target_level(input, console.print)
        with console.status("Starting the session..."):
            session = engine.submit_level(session.thread_id, level)
    question: str | None = session.question
    closing: str | None = None
    while question is not None:
        console.print(Panel(question, title="Interviewer", border_style="cyan"))
        console.print("[dim]Answer below. End with a blank line or /done.[/dim]")
        answer = read_multiline_answer(input)
        with console.status("Thinking..."):
            result = engine.submit_answer(session.thread_id, answer)
        question = result.next_question
        closing = result.closing
    if closing:
        console.print(Panel(closing, title="Interviewer", border_style="cyan"))
    console.print(f"\n[dim]{CLOSING_MESSAGE}[/dim]")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        config = resolve_model_config(
            env=os.environ,
            provider=args.provider,
            fast_model=args.fast_model,
            smart_model=args.smart_model,
        )
        posting_text = None
        if args.command == "interview" and args.job:
            posting_text = resolve_posting_text(args.job)
        callbacks = langfuse_callbacks(os.environ)
        engine = InterviewEngine(
            data_dir=_data_dir(),
            profile_parser=_build_profile_parser(config),
            assessor=_build_assessor(config),
            director=_build_director(config),
            interviewer=_build_interviewer(config),
            role_builder=_build_role_builder(config),
            researcher=_build_researcher(config),
            callbacks=callbacks,
        )
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
        CVInputError,
        PostingInputError,
        ProfileParseError,
        RoleContextError,
        AssessorError,
        DirectorError,
        EngineError,
        TracingError,
    ) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

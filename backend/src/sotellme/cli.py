import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from sotellme.config import ModelConfig, ModelConfigError, build_chat_model, resolve_model_config
from sotellme.engine import (
    EngineError,
    InterviewEngine,
    Interviewer,
    ProfileParser,
    SessionHandle,
    StarFlagger,
)
from sotellme.extraction import CVInputError
from sotellme.flagger import StarFlaggerError, flag_star_elements
from sotellme.interviewer import LLMInterviewer
from sotellme.profile import ProfileParseError, parse_candidate_profile
from sotellme.tracing import TracingError, langfuse_callbacks

CLOSING_MESSAGE = "That's a wrap. Grading and coaching arrive in a coming release."


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


def _build_star_flagger(config: ModelConfig) -> StarFlagger:
    model = build_chat_model(config, "fast")
    return lambda answer: flag_star_elements(answer, model)


def _build_interviewer(config: ModelConfig) -> Interviewer:
    return LLMInterviewer(build_chat_model(config, "fast"))


def _run_session(console: Console, engine: InterviewEngine, session: SessionHandle) -> None:
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
        callbacks = langfuse_callbacks(os.environ)
        engine = InterviewEngine(
            data_dir=_data_dir(),
            profile_parser=_build_profile_parser(config),
            star_flagger=_build_star_flagger(config),
            interviewer=_build_interviewer(config),
            callbacks=callbacks,
        )
        with engine:
            if args.command == "interview":
                with console.status("Reading your CV..."):
                    session = engine.start(Path(args.cv))
            else:
                session = engine.resume_latest()
            _run_session(console, engine, session)
    except (
        ModelConfigError,
        CVInputError,
        ProfileParseError,
        StarFlaggerError,
        EngineError,
        TracingError,
    ) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from sotellme.config import ModelConfigError, resolve_model_config
from sotellme.engine import EngineError, InterviewEngine, SessionHandle
from sotellme.extraction import CVInputError
from sotellme.tracing import TracingError, langfuse_callbacks

CLOSING_MESSAGE = "That's a wrap for the walking skeleton. Grading and coaching arrive soon."


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


def _run_question_turn(console: Console, engine: InterviewEngine, session: SessionHandle) -> None:
    console.print(Panel(session.question, title="Interviewer", border_style="cyan"))
    console.print("[dim]Answer below. End with a blank line or /done.[/dim]")
    answer = read_multiline_answer(input)
    with console.status("Wrapping up..."):
        engine.submit_answer(session.thread_id, answer)
    console.print(f"\n[green]{CLOSING_MESSAGE}[/green]")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        resolve_model_config(
            env=os.environ,
            provider=args.provider,
            fast_model=args.fast_model,
            smart_model=args.smart_model,
        )
        callbacks = langfuse_callbacks(os.environ)
        with InterviewEngine(data_dir=_data_dir(), callbacks=callbacks) as engine:
            if args.command == "interview":
                with console.status("Reading your CV..."):
                    session = engine.start(Path(args.cv))
            else:
                session = engine.resume_latest()
            _run_question_turn(console, engine, session)
    except (ModelConfigError, CVInputError, EngineError, TracingError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

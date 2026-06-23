"""Run synthetic interview simulations and judge the questions the system asks (Phase 8b).

Dev-time only. Needs a local Langfuse (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST)
and a provider key for the `run` command. Not a published command; run it from the repo with:

    uv run python scripts/simulate.py upload
    uv run python scripts/simulate.py run --persona senior-strong --persona junior-thin
    uv run python scripts/simulate.py replay --persona senior-strong

Before a run it prints an estimated cost and asks for confirmation above the gate ($3.50); pass
--yes to skip the prompt in scripted runs. `replay` re-grades and re-coaches stored sessions with
the current code via LangGraph time travel — no interviews are re-run, so it is near-free.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sotellme.catalog import CatalogError, load_catalog
from sotellme.config import ModelConfigError, resolve_model_config
from sotellme.personas import Persona, load_personas
from sotellme.sim_datasets import (
    PERSONA_DATASET_NAME,
    run_simulation_experiment,
    upload_persona_dataset,
)
from sotellme.simulation import (
    EVAL_TYPICAL_TURNS,
    confirm_run,
    estimate_run_cost,
    format_run_cost,
    replay_sessions,
)
from sotellme.tracing import TracingError, langfuse_callbacks

BACKEND = Path(__file__).resolve().parent.parent
PERSONAS_DIR = BACKEND / "evals" / "personas"
DEFAULT_MAX_TURNS = 20


def _data_dir() -> Path:
    return Path(os.environ.get("SOTELLME_DATA_DIR", "~/.sotellme")).expanduser()


def _selected_personas(names: list[str] | None) -> list[Persona]:
    personas = load_personas(PERSONAS_DIR)
    if not names:
        return personas
    by_name = {persona.name: persona for persona in personas}
    chosen = []
    for name in names:
        if name not in by_name:
            valid = ", ".join(sorted(by_name))
            raise SystemExit(f"error: unknown persona {name!r}: choose from {valid}")
        chosen.append(by_name[name])
    return chosen


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", help="LLM provider: anthropic, openai, or google_genai.")
    parser.add_argument("--fast-model", help="Override the fast model slot.")
    parser.add_argument("--smart-model", help="Override the smart model slot.")


def _upload(args: argparse.Namespace) -> int:
    personas = _selected_personas(args.persona)
    count = upload_persona_dataset(personas, os.environ)
    print(f"synced {PERSONA_DATASET_NAME}: {count} personas")
    return 0


def _run(args: argparse.Namespace) -> int:
    personas = _selected_personas(args.persona)
    config = resolve_model_config(
        env=os.environ,
        provider=args.provider,
        fast_model=args.fast_model,
        smart_model=args.smart_model,
    )
    prices = load_catalog(_data_dir()).prices

    # Price the typical session, not the --max-turns cap: real runs wrap well before the cap, so
    # estimating at the cap inflated the figure ~3.6x. The cap still bounds the worst case.
    estimate = estimate_run_cost(
        len(personas), EVAL_TYPICAL_TURNS, config.fast_model, config.smart_model, prices
    )
    print(format_run_cost(estimate))
    if not confirm_run(estimate, assume_yes=args.yes):
        print("Aborted.")
        return 1

    artifacts_dir = BACKEND / "evals" / "sessions"
    cv_dir = _data_dir() / "sim-cvs"
    print(
        run_simulation_experiment(
            config=config,
            prices=prices,
            env=os.environ,
            data_dir=_data_dir() / "sim-checkpoints",
            artifacts_dir=artifacts_dir,
            cv_dir=cv_dir,
            max_turns=args.max_turns,
            run_name=args.run_name,
            persona_names={p.name for p in personas} if args.persona else None,
        )
    )
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    print(f"\nCompare runs in Langfuse: {host} → Datasets → {PERSONA_DATASET_NAME}")
    print(f"Session artifacts written under {artifacts_dir}")
    return 0


def _replay(args: argparse.Namespace) -> int:
    from sotellme.cli import build_engine
    from sotellme.engine import InterviewEngine

    personas = _selected_personas(args.persona)
    config = resolve_model_config(
        env=os.environ,
        provider=args.provider,
        fast_model=args.fast_model,
        smart_model=args.smart_model,
    )
    prices = load_catalog(_data_dir()).prices
    artifacts_dir = BACKEND / "evals" / "sessions"
    checkpoints_dir = _data_dir() / "sim-checkpoints"

    if args.run_name:
        print(f"Replay: {args.run_name}")
    callbacks = langfuse_callbacks(os.environ)

    def make_engine(data_dir: Path) -> InterviewEngine:
        return build_engine(config, callbacks, data_dir=data_dir)

    print(
        replay_sessions(
            make_engine, personas, checkpoints_dir, artifacts_dir, prices, stage=args.stage
        )
    )
    print(f"Session artifacts refreshed under {artifacts_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simulate", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    upload = commands.add_parser("upload", help="Sync committed personas into a Langfuse dataset.")
    upload.add_argument("--persona", action="append", help="Sync only this persona (repeatable).")

    run = commands.add_parser("run", help="Run simulated interviews and judge their questions.")
    run.add_argument(
        "--persona", action="append", help="Run only this persona (repeatable); default all."
    )
    run.add_argument(
        "--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Cap questions per session."
    )
    run.add_argument("--run-name", help="Name this run; otherwise Langfuse timestamps it.")
    run.add_argument("--yes", action="store_true", help="Skip the cost-confirmation prompt.")
    _add_model_flags(run)

    replay = commands.add_parser(
        "replay",
        help="Re-grade and re-coach stored sessions via time travel; no interviews re-run.",
    )
    replay.add_argument(
        "--persona", action="append", help="Replay only this persona (repeatable); default all."
    )
    replay.add_argument(
        "--from",
        dest="stage",
        choices=["grade", "coach"],
        default="grade",
        help="Fork point: 'grade' re-runs grade and coach; 'coach' re-runs only coach.",
    )
    replay.add_argument("--run-name", help="Label this replay in the printed output.")
    _add_model_flags(replay)

    args = parser.parse_args(argv)
    try:
        if args.command == "upload":
            return _upload(args)
        if args.command == "replay":
            return _replay(args)
        return _run(args)
    except (TracingError, ModelConfigError, CatalogError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

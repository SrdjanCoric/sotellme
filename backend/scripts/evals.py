"""Sync the committed eval cases into Langfuse and run an agent over its dataset.

Dev-time only. Needs a local Langfuse (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST)
and a provider key for the `run` command. Not a published command; run it from the repo with:

    uv run python scripts/evals.py upload
    uv run python scripts/evals.py run grader
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sotellme.catalog import CatalogError, load_catalog
from sotellme.config import ModelConfigError, build_chat_model, resolve_model_config
from sotellme.eval_datasets import EvalContext, dataset_specs, run_dataset, upload_datasets
from sotellme.tracing import TracingError

BACKEND = Path(__file__).resolve().parent.parent
EVALS_DIR = BACKEND / "evals"
CTX = EvalContext(fixtures_dir=BACKEND / "tests" / "fixtures")


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", help="LLM provider: anthropic, openai, or google_genai.")
    parser.add_argument("--fast-model", help="Override the fast model slot.")
    parser.add_argument("--smart-model", help="Override the smart model slot.")


def _build_models(args: argparse.Namespace) -> dict[str, object]:
    config = resolve_model_config(
        env=os.environ,
        provider=args.provider,
        fast_model=args.fast_model,
        smart_model=args.smart_model,
    )
    return {"fast": build_chat_model(config, "fast"), "smart": build_chat_model(config, "smart")}


def _upload(args: argparse.Namespace) -> int:
    specs = list(dataset_specs().values())
    if args.agent:
        specs = [dataset_specs()[args.agent]]
    for name, count in upload_datasets(specs, EVALS_DIR, CTX, os.environ):
        print(f"synced {name}: {count} cases")
    return 0


def _run(args: argparse.Namespace) -> int:
    spec = dataset_specs()[args.agent]
    models = _build_models(args)
    data_dir = Path(os.environ.get("SOTELLME_DATA_DIR", "~/.sotellme")).expanduser()
    prices = load_catalog(data_dir).prices
    print(run_dataset(spec, models, prices, os.environ, run_name=args.run_name, limit=args.limit))
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    print(f"\nCompare runs in Langfuse: {host} → Datasets → {spec.dataset_name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    upload = commands.add_parser("upload", help="Sync committed eval cases into Langfuse datasets.")
    upload.add_argument("--agent", choices=sorted(dataset_specs()), help="Sync one agent only.")

    run = commands.add_parser("run", help="Run an agent over its dataset as a Langfuse experiment.")
    run.add_argument("agent", choices=sorted(dataset_specs()), help="Which agent to run.")
    run.add_argument("--run-name", help="Name this run; otherwise Langfuse timestamps it.")
    run.add_argument(
        "--limit", type=int, help="Run only the first N cases (calibrate cost before a full run)."
    )
    _add_model_flags(run)

    args = parser.parse_args(argv)
    try:
        if args.command == "upload":
            return _upload(args)
        return _run(args)
    except (TracingError, ModelConfigError, CatalogError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

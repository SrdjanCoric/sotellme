"""Langfuse wiring for the synthetic interview simulation (Phase 8b).

Uploads the committed personas as a Langfuse dataset and runs each one as a full simulated
interview, judging every interviewer question. Dev-time only; reuses Phase 8a's Langfuse client
and the run_experiment surface so question-quality scores compare run-to-run and slice by skill
level and answer type (carried as dataset-item metadata). Synthetic data only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler

from sotellme.budget import BudgetCallback
from sotellme.catalog import ModelPrice
from sotellme.config import ModelConfig, build_chat_model
from sotellme.eval_datasets import apply_limit, langfuse_client
from sotellme.judge import QuestionJudge
from sotellme.personas import Persona
from sotellme.pricing import format_cost_summary, summarize_actual_cost
from sotellme.simulation import (
    judge_session,
    run_persona_simulation,
    write_session_artifact,
)
from sotellme.simulator import CandidateSimulator

if TYPE_CHECKING:
    from langfuse import Langfuse

PERSONA_DATASET_NAME = "sotellme-personas"

PERSONA_DATASET_DESCRIPTION = (
    "Synthetic interview personas (evals/personas), one full simulated interview per item; "
    "synthetic data only."
)

_COVERAGE_SCORE = {"good": 1.0, "weak": 0.5, "bad": 0.0}


def _persona_metadata(persona: Persona) -> dict[str, Any]:
    behaviors = sorted(
        {persona.base_behavior, *(planted.behavior for planted in persona.planted_turns)}
    )
    return {
        "target_level": persona.target_level,
        "base_behavior": persona.base_behavior,
        "behaviors": behaviors,
    }


def upload_persona_dataset(
    personas: Sequence[Persona],
    env: Mapping[str, str],
    dataset_name: str = PERSONA_DATASET_NAME,
) -> int:
    client = langfuse_client(env)
    client.create_dataset(name=dataset_name, description=PERSONA_DATASET_DESCRIPTION)
    for persona in personas:
        client.create_dataset_item(
            dataset_name=dataset_name,
            input=persona.model_dump(),
            metadata=_persona_metadata(persona),
            id=f"{dataset_name}:{persona.name}",
        )
    return len(personas)


def select_persona_items(items: Sequence[Any], persona_names: set[str] | None) -> list[Any]:
    """Keep only the dataset items for the named personas, so `--persona` limits what runs (and
    is billed), not just the cost estimate. None means run them all."""
    if not persona_names:
        return list(items)
    return [item for item in items if item.input.get("name") in persona_names]


def run_simulation_experiment(
    config: ModelConfig,
    prices: Mapping[str, ModelPrice],
    env: Mapping[str, str],
    data_dir: Path,
    artifacts_dir: Path,
    cv_dir: Path,
    max_turns: int,
    run_name: str | None = None,
    limit: int | None = None,
    persona_names: set[str] | None = None,
    dataset_name: str = PERSONA_DATASET_NAME,
) -> str:
    from langfuse import Evaluation

    client: Langfuse = langfuse_client(env)
    budget = BudgetCallback()
    callbacks: list[BaseCallbackHandler] = [budget]

    simulator_model = build_chat_model(config, "fast")
    simulator_model.callbacks = [budget]
    simulator = CandidateSimulator(simulator_model, config.provider)

    judge_model = build_chat_model(config, "smart")
    judge_model.callbacks = [budget]
    judge = QuestionJudge(judge_model, config.provider)

    dataset = client.get_dataset(dataset_name)
    items = apply_limit(select_persona_items(dataset.items, persona_names), limit)

    def task(*, item: Any, **_: Any) -> dict[str, Any]:
        persona = Persona.model_validate(item.input)
        session = run_persona_simulation(
            persona, simulator, config, callbacks, data_dir, cv_dir, max_turns
        )
        write_session_artifact(session, artifacts_dir)
        judgement = judge_session(judge, session)
        return {"session": session.model_dump(), "judgement": judgement.model_dump()}

    def evaluator(*, output: Any, **_: Any) -> list[Evaluation]:
        judgement = output["judgement"]
        scores = [
            Evaluation(name=dimension, value=mean)
            for dimension, mean in judgement["dimension_means"].items()
        ]
        coverage = judgement["coverage"]
        scores.append(
            Evaluation(
                name="competency_coverage",
                value=_COVERAGE_SCORE[coverage["verdict"]],
                comment=coverage["rationale"],
            )
        )
        return scores

    result = client.run_experiment(
        name=dataset_name,
        run_name=run_name,
        data=items,
        task=task,
        evaluators=[evaluator],
    )
    client.flush()
    cost = format_cost_summary(summarize_actual_cost(budget.usage, prices))
    return f"{result.format(include_item_results=True)}\n\n{cost}"

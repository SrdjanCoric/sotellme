"""Langfuse wiring for the synthetic interview simulation.

Uploads the committed personas as a Langfuse dataset and runs each one as a full simulated
interview, judging every interviewer question. Dev-time only; reuses the Langfuse client and
the run_experiment surface so question-quality scores compare run-to-run and slice by skill
level and answer type (carried as dataset-item metadata). Synthetic data only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, TypedDict

from langchain_core.callbacks import BaseCallbackHandler

from sotellme.budget import BudgetCallback
from sotellme.catalog import ModelPrice
from sotellme.config import ModelConfig, build_chat_model
from sotellme.eval_datasets import apply_limit, langfuse_client
from sotellme.eval_progress import EvalProgress
from sotellme.grader import GradingError
from sotellme.judge import JudgeError, QuestionJudge
from sotellme.personas import AnswerBehavior, Persona
from sotellme.pricing import ModelUsage, format_cost_summary, merge_usage, summarize_actual_cost
from sotellme.role import TargetLevel
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


class PersonaMetadata(TypedDict):
    """Dataset-item metadata carried with each persona for slicing scores."""

    target_level: TargetLevel
    base_behavior: AnswerBehavior
    behaviors: list[AnswerBehavior]


def _persona_metadata(persona: Persona) -> PersonaMetadata:
    """Derive the slicing metadata for a persona from its base and planted behaviors."""
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
    """Create the persona Langfuse dataset and upload each persona as an item.

    Args:
        personas: The personas to upload.
        env: Environment mapping holding the Langfuse keys.
        dataset_name: Name of the dataset to create and upload into.

    Returns:
        The number of personas uploaded.

    Raises:
        TracingError: If the Langfuse keys are unset or the langfuse package is missing.
    """
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
    """Keep only the dataset items for the named personas so `--persona` limits what runs.

    Limits what is billed, not just the cost estimate. None means run them all.
    """
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
    """Run each persona as a full simulated interview and judge it as a Langfuse experiment.

    Builds the simulator and judge models on the fast and smart slots, fetches and limits
    the persona dataset, then for each item runs the simulation, writes its artifact,
    judges it, and emits per-dimension scores plus a competency-coverage score. Appends a
    cost summary.

    Args:
        config: The model configuration for the simulation's agents.
        prices: Model price lookup used to summarize cost.
        env: Environment mapping holding the Langfuse keys.
        data_dir: Directory the engine persists session data into.
        artifacts_dir: Directory the session artifacts are written into.
        cv_dir: Directory persona CVs are written into.
        max_turns: Maximum number of turns per simulated session.
        run_name: Optional name for this experiment run.
        limit: Optional cap on how many personas to run.
        persona_names: Persona names to run; None runs all.
        dataset_name: Name of the persona dataset to run.

    Returns:
        The formatted experiment result followed by a cost summary.

    Raises:
        TracingError: If the Langfuse keys are unset or the langfuse package is missing.
    """
    from langfuse import Evaluation

    client: Langfuse = langfuse_client(env)

    dataset = client.get_dataset(dataset_name)
    items = apply_limit(select_persona_items(dataset.items, persona_names), limit)

    # Personas run concurrently (independent interviews), so each gets its own budget for an
    # exact per-persona cost; a lock guards the running total and the per-persona usage we fold
    # into the run's grand total at the end.
    progress = EvalProgress(len(items))
    order = {item.input["name"]: position for position, item in enumerate(items, start=1)}
    cost_lock = Lock()
    running_usd = [0.0]
    collected_usage: list[dict[str, ModelUsage]] = []

    def task(*, item: Any, **_: Any) -> dict[str, Any]:
        persona = Persona.model_validate(item.input)
        index = order[persona.name]
        progress.start(index, persona.name)

        persona_budget = BudgetCallback()
        persona_callbacks: list[BaseCallbackHandler] = [persona_budget]
        simulator_model = build_chat_model(config, "fast")
        simulator_model.callbacks = persona_callbacks
        simulator = CandidateSimulator(simulator_model, config.provider)
        judge_model = build_chat_model(config, "smart")
        judge_model.callbacks = persona_callbacks
        judge = QuestionJudge(judge_model, config.provider)

        try:
            session = run_persona_simulation(
                persona, simulator, config, persona_callbacks, data_dir, cv_dir, max_turns
            )
        except GradingError as exc:
            raise GradingError(exc.diagnostic()) from exc
        write_session_artifact(session, artifacts_dir)
        try:
            judgement = judge_session(
                judge, session, expected_to_terminate=persona.expected_to_terminate
            )
        except JudgeError as exc:
            raise JudgeError(exc.diagnostic()) from exc

        persona_usd = summarize_actual_cost(persona_budget.usage, prices).usd
        with cost_lock:
            running_usd[0] += persona_usd
            total_usd = running_usd[0]
            collected_usage.append(persona_budget.usage)
        progress.finish(
            index,
            persona.name,
            turns=session.turns,
            finished_reason=session.finished_reason,
            persona_usd=persona_usd,
            total_usd=total_usd,
        )
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
    cost = format_cost_summary(summarize_actual_cost(merge_usage(collected_usage), prices))
    return f"{result.format(include_item_results=True)}\n\n{cost}"

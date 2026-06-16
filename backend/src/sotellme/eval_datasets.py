from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models import BaseChatModel

from sotellme.assessor import assess_answer
from sotellme.budget import BudgetCallback
from sotellme.catalog import ModelPrice
from sotellme.coach import CoachReport, coach_session
from sotellme.extraction import extract_cv_text
from sotellme.grader import AnswerScore, SessionGrade, grade_session
from sotellme.interviewer import Turn
from sotellme.pricing import format_cost_summary, summarize_actual_cost
from sotellme.profile import CandidateProfile, parse_candidate_profile
from sotellme.role import RoleContext, build_role_context
from sotellme.tracing import TracingError
from sotellme.voice import voice_tells

if TYPE_CHECKING:
    from langfuse import Langfuse

STAR_ELEMENTS = ("situation", "task", "action", "result", "quantified_result")


@dataclass(frozen=True)
class EvalContext:
    fixtures_dir: Path


@dataclass(frozen=True)
class DatasetItem:
    id: str
    input: dict[str, Any]
    expected_output: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalScore:
    name: str
    value: float
    comment: str = ""


def _turns_from(transcript: Sequence[dict[str, Any]]) -> list[Turn]:
    return [Turn(question=t["question"], answer=t["answer"]) for t in transcript]


def disagreements(answer: AnswerScore, expected: dict[str, Any]) -> dict[str, Any]:
    observed = answer.star.model_dump()
    found: dict[str, Any] = {}
    star_misses = {
        flag: observed[flag] for flag in STAR_ELEMENTS if observed[flag] != expected["star"][flag]
    }
    if star_misses:
        found["star"] = star_misses
    required = set(expected["weak_or_missing"])
    present = {flag for flag in STAR_ELEMENTS if expected["star"][flag]}
    allowed = required | present
    got = set(answer.weak_or_missing)
    if not (required <= got <= allowed):
        found["weak_or_missing"] = {"got": sorted(got), "want_at_least": sorted(required)}
    if answer.specificity != expected["specificity"]:
        found["specificity"] = {"got": answer.specificity, "want": expected["specificity"]}
    if answer.ownership != expected["ownership"]:
        found["ownership"] = {"got": answer.ownership, "want": expected["ownership"]}
    if abs(answer.score - expected["score"]) > 1:
        found["score"] = {"got": answer.score, "want": expected["score"]}
    return found


def _profile_field_texts(profile: CandidateProfile, field_name: str) -> list[str]:
    if field_name == "roles":
        return [f"{r.title} {r.organization} {r.period or ''}" for r in profile.roles]
    if field_name == "projects":
        return [f"{p.name} {p.description}" for p in profile.projects]
    if field_name == "quantified_claims":
        return profile.quantified_claims
    if field_name == "technologies":
        return profile.technologies
    raise ValueError(f"unknown profile field in eval case: {field_name}")


def _role_failures(context: RoleContext, expect: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    company = (context.company or "").lower()
    framework = (context.framework or "").lower()
    names = [c.name.lower() for c in context.competencies]

    if "company_contains" in expect and expect["company_contains"] not in company:
        failures.append(f"company {context.company!r} lacks {expect['company_contains']!r}")
    if "company_not_contains" in expect and expect["company_not_contains"] in company:
        failures.append(f"company {context.company!r} obeyed the injected instruction")
    if expect.get("framework_null") and context.framework is not None:
        failures.append(f"expected no framework, got {context.framework!r}")
    if "framework_contains" in expect and expect["framework_contains"] not in framework:
        failures.append(f"framework {context.framework!r} lacks {expect['framework_contains']!r}")
    if "framework_not_contains" in expect and expect["framework_not_contains"] in framework:
        failures.append(f"framework {context.framework!r} obeyed the injected instruction")
    if "competencies_include_any" in expect:
        included = [
            wanted
            for wanted in expect["competencies_include_any"]
            if any(wanted in name for name in names)
        ]
        if len(included) < expect.get("min_included", 1):
            failures.append(f"competencies {names} include too few of the framework principles")
    if "competencies_within" in expect:
        allowed = set(expect["competencies_within"])
        strays = [name for name in names if name not in allowed]
        if strays:
            failures.append(f"competencies stray from the default map: {strays}")
    if "target_level" in expect and context.target_level != expect["target_level"]:
        failures.append(f"level {context.target_level!r} != {expect['target_level']!r}")
    if expect.get("target_level_null") and context.target_level is not None:
        failures.append(f"expected no deduced level, got {context.target_level!r}")
    return failures


def _transcript_agreement(scores: Sequence[AnswerScore], expected: dict[str, Any]) -> EvalScore:
    too_low = [i for i in expected.get("senior_floor_turns", []) if scores[i - 1].score < 4]
    too_high = [i for i in expected.get("senior_ceiling_turns", []) if scores[i - 1].score > 3]
    parts = []
    if too_low:
        parts.append(f"under-levelled at turns {too_low}")
    if too_high:
        parts.append(f"over-levelled at turns {too_high}")
    return EvalScore(
        name="grader_agreement",
        value=0.0 if parts else 1.0,
        comment="; ".join(parts),
    )


class AgentEval(ABC):
    agent: str
    dataset_name: str
    cases_file: str
    model_slot: str

    @abstractmethod
    def to_input(self, case: dict[str, Any], ctx: EvalContext) -> dict[str, Any]: ...

    @abstractmethod
    def to_expected(self, case: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def evaluate(
        self, output: dict[str, Any], expected: dict[str, Any], metadata: dict[str, Any]
    ) -> list[EvalScore]: ...

    @abstractmethod
    def run(self, item_input: dict[str, Any], model: BaseChatModel) -> dict[str, Any]: ...

    def to_metadata(self, case: dict[str, Any]) -> dict[str, Any]:
        return {}

    def to_items(self, cases: Sequence[dict[str, Any]], ctx: EvalContext) -> list[DatasetItem]:
        return [
            DatasetItem(
                id=f"{self.dataset_name}:{case['name']}",
                input=self.to_input(case, ctx),
                expected_output=self.to_expected(case),
                metadata=self.to_metadata(case),
            )
            for case in cases
        ]


class GraderEval(AgentEval):
    agent = "grader"
    dataset_name = "sotellme-grader"
    cases_file = "grader_cases.json"
    model_slot = "smart"

    def to_input(self, case: dict[str, Any], ctx: EvalContext) -> dict[str, Any]:
        turns = (
            case["turns"]
            if "turns" in case
            else [{"question": case["question"], "answer": case["answer"]}]
        )
        return {"turns": turns, "target_level": case["target_level"]}

    def to_expected(self, case: dict[str, Any]) -> dict[str, Any]:
        if "proposed" in case:
            return cast(dict[str, Any], case["proposed"])
        return {
            "senior_floor_turns": case.get("senior_floor_turns", []),
            "senior_ceiling_turns": case.get("senior_ceiling_turns", []),
        }

    def to_metadata(self, case: dict[str, Any]) -> dict[str, Any]:
        return {"kind": "single" if "proposed" in case else "transcript"}

    def evaluate(
        self, output: dict[str, Any], expected: dict[str, Any], metadata: dict[str, Any]
    ) -> list[EvalScore]:
        scores = [AnswerScore.model_validate(score) for score in output["scores"]]
        if metadata.get("kind") == "transcript":
            return [_transcript_agreement(scores, expected)]
        found = disagreements(scores[0], expected)
        comment = "; ".join(f"{key}: {value}" for key, value in found.items())
        return [EvalScore(name="grader_agreement", value=0.0 if found else 1.0, comment=comment)]

    def run(self, item_input: dict[str, Any], model: BaseChatModel) -> dict[str, Any]:
        turns = _turns_from(item_input["turns"])
        return grade_session(turns, item_input["target_level"], model).model_dump()


class AssessorEval(AgentEval):
    agent = "assessor"
    dataset_name = "sotellme-assessor"
    cases_file = "assessor_cases.json"
    model_slot = "fast"

    def to_input(self, case: dict[str, Any], ctx: EvalContext) -> dict[str, Any]:
        return {"topic": case["topic"], "answer": case["answer"]}

    def to_expected(self, case: dict[str, Any]) -> dict[str, Any]:
        return {**case["expected"], "claim_substrings": case.get("claim_substrings", [])}

    def evaluate(
        self, output: dict[str, Any], expected: dict[str, Any], metadata: dict[str, Any]
    ) -> list[EvalScore]:
        observed = {**output["star"], "sufficient_signal": output["sufficient_signal"]}
        flags = (*STAR_ELEMENTS, "sufficient_signal")
        misread = {flag: observed[flag] for flag in flags if observed[flag] != expected[flag]}
        chased = " | ".join(output["claims_worth_chasing"]).lower()
        missing = [s for s in expected["claim_substrings"] if s.lower() not in chased]
        parts = []
        if misread:
            parts.append(f"misread {misread}")
        if missing:
            parts.append(f"missed claims {missing}")
        return [
            EvalScore(
                name="assessor_agreement",
                value=0.0 if parts else 1.0,
                comment="; ".join(parts),
            )
        ]

    def run(self, item_input: dict[str, Any], model: BaseChatModel) -> dict[str, Any]:
        topic = item_input["topic"]
        turn = Turn(question=f"Tell me about {topic}.", answer=item_input["answer"])
        return assess_answer(topic, [turn], model).model_dump()


class RoleEval(AgentEval):
    agent = "role"
    dataset_name = "sotellme-role-context"
    cases_file = "role_context_cases.json"
    model_slot = "fast"

    def to_input(self, case: dict[str, Any], ctx: EvalContext) -> dict[str, Any]:
        return {"posting": case["posting"]}

    def to_expected(self, case: dict[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], case["expect"])

    def evaluate(
        self, output: dict[str, Any], expected: dict[str, Any], metadata: dict[str, Any]
    ) -> list[EvalScore]:
        failures = _role_failures(RoleContext.model_validate(output), expected)
        return [
            EvalScore(
                name="role_agreement",
                value=0.0 if failures else 1.0,
                comment="; ".join(failures),
            )
        ]

    def run(self, item_input: dict[str, Any], model: BaseChatModel) -> dict[str, Any]:
        return build_role_context(item_input["posting"], model).model_dump()


class ProfileEval(AgentEval):
    agent = "profile"
    dataset_name = "sotellme-profile-parser"
    cases_file = "profile_parser_cases.json"
    model_slot = "fast"

    def to_input(self, case: dict[str, Any], ctx: EvalContext) -> dict[str, Any]:
        return {"cv_text": extract_cv_text(ctx.fixtures_dir / case["fixture"])}

    def to_expected(self, case: dict[str, Any]) -> dict[str, Any]:
        return {"named_facts": case["named_facts"]}

    def evaluate(
        self, output: dict[str, Any], expected: dict[str, Any], metadata: dict[str, Any]
    ) -> list[EvalScore]:
        profile = CandidateProfile.model_validate(output)
        facts = expected["named_facts"]
        missing = [
            fact["substring"]
            for fact in facts
            if not any(
                fact["substring"].lower() in text.lower()
                for text in _profile_field_texts(profile, fact["field"])
            )
        ]
        surfaced = len(facts) - len(missing)
        return [
            EvalScore(
                name="facts_surfaced",
                value=surfaced / len(facts) if facts else 1.0,
                comment=f"not surfaced: {missing}" if missing else "",
            )
        ]

    def run(self, item_input: dict[str, Any], model: BaseChatModel) -> dict[str, Any]:
        return parse_candidate_profile(item_input["cv_text"], model).model_dump()


def _coaching_prose(report: CoachReport) -> str:
    parts = [report.summary, report.study_plan]
    for advice in report.answer_advice:
        parts.extend([advice.diagnosis, advice.fix])
    for drill in report.drills:
        parts.extend([drill.focus, drill.exercise])
    return "\n".join(parts)


class CoachEval(AgentEval):
    agent = "coach"
    dataset_name = "sotellme-coach"
    cases_file = "coach_cases.json"
    model_slot = "smart"

    def to_input(self, case: dict[str, Any], ctx: EvalContext) -> dict[str, Any]:
        return {
            "transcript": case["transcript"],
            "grade": case["grade"],
            "target_level": case["target_level"],
        }

    def to_expected(self, case: dict[str, Any]) -> dict[str, Any]:
        return {"gap_summary": case["gap_summary"]}

    def evaluate(
        self, output: dict[str, Any], expected: dict[str, Any], metadata: dict[str, Any]
    ) -> list[EvalScore]:
        tells = voice_tells(_coaching_prose(CoachReport.model_validate(output)))
        return [
            EvalScore(
                name="coach_voice",
                value=0.0 if tells else 1.0,
                comment=", ".join(tells),
            )
        ]

    def run(self, item_input: dict[str, Any], model: BaseChatModel) -> dict[str, Any]:
        turns = _turns_from(item_input["transcript"])
        grade = SessionGrade.model_validate(item_input["grade"])
        return coach_session(turns, grade, item_input["target_level"], model).model_dump()


def dataset_specs() -> dict[str, AgentEval]:
    specs: list[AgentEval] = [
        GraderEval(),
        AssessorEval(),
        RoleEval(),
        ProfileEval(),
        CoachEval(),
    ]
    return {spec.agent: spec for spec in specs}


def build_items(spec: AgentEval, evals_dir: Path, ctx: EvalContext) -> list[DatasetItem]:
    document = json.loads((evals_dir / spec.cases_file).read_text())
    cases: list[dict[str, Any]] = document["cases"]
    return spec.to_items(cases, ctx)


DATASET_DESCRIPTION = "Synthetic sotellme eval cases, synced from evals/{file} (source of truth)."

DEFAULT_LANGFUSE_TIMEOUT = 30


def _resolve_timeout(env: Mapping[str, str]) -> int:
    try:
        return int(env["LANGFUSE_TIMEOUT"])
    except (KeyError, ValueError):
        return DEFAULT_LANGFUSE_TIMEOUT


def _langfuse_client(env: Mapping[str, str]) -> Langfuse:
    if not (env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY")):
        raise TracingError(
            "Langfuse keys are not set. Export LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, "
            "and LANGFUSE_HOST for your local instance before running evals."
        )
    try:
        from langfuse import Langfuse
    except ModuleNotFoundError as missing:
        raise TracingError(
            "The langfuse package is not installed. Install it with: uv sync --extra tracing"
        ) from missing
    return Langfuse(timeout=_resolve_timeout(env))


def upload_datasets(
    specs: Sequence[AgentEval],
    evals_dir: Path,
    ctx: EvalContext,
    env: Mapping[str, str],
) -> list[tuple[str, int]]:
    client = _langfuse_client(env)
    uploaded: list[tuple[str, int]] = []
    for spec in specs:
        client.create_dataset(
            name=spec.dataset_name,
            description=DATASET_DESCRIPTION.format(file=spec.cases_file),
        )
        items = build_items(spec, evals_dir, ctx)
        for item in items:
            client.create_dataset_item(
                dataset_name=spec.dataset_name,
                input=item.input,
                expected_output=item.expected_output,
                metadata=item.metadata,
                id=item.id,
            )
        uploaded.append((spec.dataset_name, len(items)))
    return uploaded


def run_dataset(
    spec: AgentEval,
    models: Mapping[str, BaseChatModel],
    prices: Mapping[str, ModelPrice],
    env: Mapping[str, str],
    run_name: str | None = None,
    limit: int | None = None,
) -> str:
    from langfuse import Evaluation

    client = _langfuse_client(env)
    budget = BudgetCallback()
    model = models[spec.model_slot]
    model.callbacks = [budget]
    dataset = client.get_dataset(spec.dataset_name)
    items = list(dataset.items)[:limit] if limit else dataset.items

    def task(*, item: Any, **_: Any) -> dict[str, Any]:
        return spec.run(item.input, model)

    def evaluator(
        *, input: Any, output: Any, expected_output: Any = None, metadata: Any = None, **_: Any
    ) -> list[Evaluation]:
        scores = spec.evaluate(output, expected_output or {}, metadata or {})
        return [Evaluation(name=s.name, value=s.value, comment=s.comment) for s in scores]

    result = client.run_experiment(
        name=spec.dataset_name,
        run_name=run_name,
        data=items,
        task=task,
        evaluators=[evaluator],
    )
    client.flush()
    cost = format_cost_summary(summarize_actual_cost(budget.usage, prices))
    return f"{result.format(include_item_results=True)}\n\n{cost}"

import json
from pathlib import Path
from typing import Any, cast

from sotellme.assessor import StarFlags
from sotellme.eval_datasets import (
    DEFAULT_LANGFUSE_TIMEOUT,
    EvalContext,
    _resolve_timeout,
    apply_limit,
    build_items,
    dataset_specs,
)
from sotellme.grader import AnswerScore, SessionGrade, SkippedTurn


def _turn_score(turn_index: int, score: int) -> AnswerScore:
    return AnswerScore(
        question=f"Turn {turn_index} question.",
        turn_index=turn_index,
        star=StarFlags(situation=True, task=True, action=True, result=True, quantified_result=True),
        specificity="high",
        ownership="clear",
        weak_or_missing=[],
        gap="" if score == 5 else "One refinement short of a five.",
        rationale="A complete, quantified story at the target level.",
        score=score,
    )


EVALS_DIR = Path(__file__).parent.parent / "evals"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
CTX = EvalContext(fixtures_dir=FIXTURES_DIR)


def test_resolve_timeout_honors_a_valid_override() -> None:
    assert _resolve_timeout({"LANGFUSE_TIMEOUT": "60"}) == 60


def test_resolve_timeout_defaults_when_unset() -> None:
    assert _resolve_timeout({}) == DEFAULT_LANGFUSE_TIMEOUT


def test_resolve_timeout_falls_back_on_a_non_integer() -> None:
    assert _resolve_timeout({"LANGFUSE_TIMEOUT": "soon"}) == DEFAULT_LANGFUSE_TIMEOUT


def test_apply_limit_of_zero_selects_nothing() -> None:
    assert apply_limit([1, 2, 3], 0) == []


def test_apply_limit_of_none_keeps_every_item() -> None:
    assert apply_limit([1, 2, 3], None) == [1, 2, 3]


def test_apply_limit_caps_to_the_requested_count() -> None:
    assert apply_limit([1, 2, 3], 2) == [1, 2]


def test_every_committed_case_file_maps_through_its_spec() -> None:
    for spec in dataset_specs().values():
        items = build_items(spec, EVALS_DIR, CTX)
        assert items, f"{spec.agent}: no cases loaded"
        assert all(item.id.startswith(f"{spec.dataset_name}:") for item in items)
        assert all(item.input for item in items)


def test_grader_single_answer_case_maps_to_a_labelled_item() -> None:
    spec = dataset_specs()["grader"]
    case = {
        "name": "complete-quantified-single-team-senior",
        "question": "Tell me about a project you're proud of.",
        "answer": "I led the migration to a streaming pipeline and cut latency to 90 seconds.",
        "target_level": "senior",
        "proposed": {
            "star": {
                "situation": True,
                "task": True,
                "action": True,
                "result": True,
                "quantified_result": True,
            },
            "specificity": "high",
            "ownership": "clear",
            "weak_or_missing": [],
            "score": 4,
        },
        "note": "solid senior",
    }

    [item] = spec.to_items([case], CTX)

    assert item.input["target_level"] == "senior"
    assert item.input["turns"] == [{"question": case["question"], "answer": case["answer"]}]
    assert item.expected_output == case["proposed"]
    assert item.id == "sotellme-grader:complete-quantified-single-team-senior"


def test_grader_transcript_agreement_maps_scores_by_turn_index_past_a_skip() -> None:
    spec = dataset_specs()["grader"]
    grade = SessionGrade(
        scores=[_turn_score(1, 3), _turn_score(3, 4)],
        skipped=[SkippedTurn(turn_index=2, question="First or second?", reason="Clarifying.")],
    )
    expected = {"senior_floor_turns": [3], "senior_ceiling_turns": [1]}

    [result] = spec.evaluate(grade.model_dump(), expected, {"kind": "transcript"})

    assert result.value == 1.0


def test_grader_transcript_agreement_flags_a_scored_clarifying_turn() -> None:
    spec = dataset_specs()["grader"]
    grade = SessionGrade(scores=[_turn_score(1, 4), _turn_score(2, 3)])
    expected = {"senior_floor_turns": [1], "clarifying_turns": [2]}

    [result] = spec.evaluate(grade.model_dump(), expected, {"kind": "transcript"})

    assert result.value == 0.0
    assert "clarifying" in result.comment


def test_grader_transcript_agreement_flags_a_turn_scored_twice() -> None:
    spec = dataset_specs()["grader"]
    grade = SessionGrade(scores=[_turn_score(2, 4), _turn_score(2, 4)])
    expected = {"senior_floor_turns": [2]}

    [result] = spec.evaluate(grade.model_dump(), expected, {"kind": "transcript"})

    assert result.value == 0.0
    assert "duplicate" in result.comment


def _grader_cases() -> list[dict[str, Any]]:
    document = json.loads((EVALS_DIR / "grader_cases.json").read_text())
    return cast(list[dict[str, Any]], document["cases"])


def _single_answer_cases() -> list[dict[str, Any]]:
    return [case for case in _grader_cases() if "proposed" in case]


def _case_named(name: str) -> dict[str, Any]:
    [case] = [case for case in _grader_cases() if case["name"] == name]
    return case


def test_committed_grader_cases_honor_the_gap_invariant() -> None:
    for case in _single_answer_cases():
        proposed = case["proposed"]
        has_gap = bool(proposed.get("gap", "").strip())
        assert has_gap == (proposed["score"] < 5), (
            f"{case['name']}: a gap is non-empty exactly when the score is below 5"
        )


def test_under_leveled_senior_case_carries_a_visible_leveling_gap() -> None:
    proposed = _case_named("complete-quantified-single-team-senior")["proposed"]

    assert proposed["score"] == 4
    assert proposed["gap"].strip()


def test_grader_cases_pin_the_four_versus_three_boundary() -> None:
    refinement = _case_named("complete-quantified-single-team-senior")["proposed"]
    real_miss = _case_named("all-impact-no-difficulty-senior")["proposed"]

    assert refinement["score"] == 4
    assert real_miss["score"] == 3
    assert refinement["gap"].strip() and real_miss["gap"].strip()


def test_small_team_deference_case_is_not_docked_on_ownership() -> None:
    proposed = _case_named("small-team-deference-owns-execution-senior")["proposed"]

    assert proposed["ownership"] == "clear"
    assert proposed["score"] >= 4


def test_confidentiality_case_judges_specificity_on_what_was_described() -> None:
    proposed = _case_named("confidentiality-declined-but-general-senior")["proposed"]

    assert proposed["specificity"] in {"medium", "high"}
    assert proposed["score"] >= 3


def test_clarifying_turn_case_pins_a_skipped_turn_disjoint_from_the_scored_ones() -> None:
    case = _case_named("clarifying-turn-skipped-senior-session")
    clarifying = set(case["clarifying_turns"])
    scored = set(case.get("senior_floor_turns", [])) | set(case.get("senior_ceiling_turns", []))
    turn_count = len(case["turns"])

    assert clarifying
    assert all(1 <= turn <= turn_count for turn in clarifying)
    assert not (clarifying & scored), "a turn cannot be both skipped and scored"


def test_clarifying_turns_map_through_the_spec_into_the_expectation() -> None:
    spec = dataset_specs()["grader"]
    case = _case_named("clarifying-turn-skipped-senior-session")

    [item] = spec.to_items([case], CTX)

    assert item.expected_output["clarifying_turns"] == case["clarifying_turns"]


def test_a_case_label_never_leaks_into_the_dataset_input() -> None:
    spec = dataset_specs()["grader"]
    case = {
        "name": "x",
        "question": "q",
        "answer": "a",
        "target_level": "mid",
        "proposed": {"score": 2},
    }

    [item] = spec.to_items([case], CTX)

    assert "proposed" not in item.input
    assert "score" not in item.input


def _answer_score(score: int, weak: list[str] | None = None) -> dict[str, Any]:
    return {
        "question": "Tell me about a project you're proud of.",
        "turn_index": 1,
        "rationale": "r",
        "star": {
            "situation": True,
            "task": True,
            "action": True,
            "result": not weak,
            "quantified_result": not weak,
        },
        "specificity": "high",
        "ownership": "clear",
        "weak_or_missing": weak or [],
        "gap": "" if score == 5 else "g",
        "score": score,
    }


GRADER_LABEL = {
    "star": {
        "situation": True,
        "task": True,
        "action": True,
        "result": True,
        "quantified_result": True,
    },
    "specificity": "high",
    "ownership": "clear",
    "weak_or_missing": [],
    "score": 4,
}


def test_grader_evaluator_agrees_with_a_matching_grade() -> None:
    spec = dataset_specs()["grader"]
    output = {"scores": [_answer_score(4)]}

    [result] = spec.evaluate(output, GRADER_LABEL, {"kind": "single"})

    assert result.name == "grader_agreement"
    assert result.value == 1.0


def test_grader_evaluator_flags_a_grade_that_disagrees() -> None:
    spec = dataset_specs()["grader"]
    output = {"scores": [_answer_score(1, weak=["result", "quantified_result"])]}

    [result] = spec.evaluate(output, GRADER_LABEL, {"kind": "single"})

    assert result.value == 0.0
    assert result.comment


def test_assessor_case_maps_topic_and_answer_to_input() -> None:
    spec = dataset_specs()["assessor"]
    case = {
        "name": "complete-and-quantified",
        "topic": "the dashboard latency work",
        "answer": "I led the migration and cut latency to 90 seconds.",
        "expected": {
            "situation": True,
            "task": True,
            "action": True,
            "result": True,
            "quantified_result": True,
            "sufficient_signal": True,
        },
    }

    [item] = spec.to_items([case], CTX)

    assert item.input == {"topic": case["topic"], "answer": case["answer"]}
    assert item.expected_output["situation"] is True


def test_assessor_evaluator_rewards_a_correct_reading_and_flags_a_misread() -> None:
    spec = dataset_specs()["assessor"]
    expected = {
        "situation": True,
        "task": True,
        "action": True,
        "result": True,
        "quantified_result": True,
        "sufficient_signal": True,
        "claim_substrings": ["90 seconds"],
    }
    star = {
        "situation": True,
        "task": True,
        "action": True,
        "result": True,
        "quantified_result": True,
    }

    correct = {"star": star, "sufficient_signal": True, "claims_worth_chasing": ["the 90 seconds"]}
    [hit] = spec.evaluate(correct, expected, {})
    assert hit.name == "assessor_agreement"
    assert hit.value == 1.0

    misread = {
        "star": {**star, "quantified_result": False},
        "sufficient_signal": False,
        "claims_worth_chasing": [],
    }
    [miss] = spec.evaluate(misread, expected, {})
    assert miss.value == 0.0
    assert miss.comment


def _role_output(company: str) -> dict[str, Any]:
    return {
        "company": company,
        "framework": "leadership principles",
        "competencies": [
            {"name": "ownership", "weight": 5},
            {"name": "customer obsession", "weight": 4},
        ],
        "target_level": "senior",
    }


def test_role_evaluator_checks_the_expectations_block() -> None:
    spec = dataset_specs()["role"]
    expect = {
        "company_contains": "amazon",
        "framework_contains": "leadership principles",
        "competencies_include_any": ["ownership"],
        "target_level": "senior",
    }

    [hit] = spec.evaluate(_role_output("Amazon"), expect, {})
    assert hit.name == "role_agreement"
    assert hit.value == 1.0

    [miss] = spec.evaluate(_role_output("Globex"), expect, {})
    assert miss.value == 0.0
    assert miss.comment


def test_profile_case_extracts_the_cv_text_at_upload() -> None:
    spec = dataset_specs()["profile"]
    case = {
        "name": "markdown-cv",
        "fixture": "synthetic_cv.md",
        "named_facts": [{"field": "technologies", "substring": "Kafka"}],
    }

    [item] = spec.to_items([case], CTX)

    assert "Helioscope" in item.input["cv_text"]
    assert "fixture" not in item.input["cv_text"]
    assert item.expected_output["named_facts"] == case["named_facts"]


def test_profile_evaluator_scores_the_fraction_of_facts_surfaced() -> None:
    spec = dataset_specs()["profile"]
    output = {
        "roles": [{"title": "Senior Software Engineer", "organization": "Helioscope Analytics"}],
        "projects": [],
        "quantified_claims": ["cut latency 38%"],
        "technologies": ["Kafka", "Python"],
    }
    expected = {
        "named_facts": [
            {"field": "technologies", "substring": "Kafka"},
            {"field": "roles", "substring": "Helioscope Analytics"},
            {"field": "quantified_claims", "substring": "missing-number"},
        ]
    }

    [result] = spec.evaluate(output, expected, {})

    assert result.name == "facts_surfaced"
    assert result.value == 2 / 3
    assert "missing-number" in result.comment


def test_coach_case_maps_transcript_grade_and_level() -> None:
    spec = dataset_specs()["coach"]
    case = {
        "name": "missing-quantified-result-senior",
        "transcript": [{"question": "q", "answer": "a"}],
        "grade": {"scores": []},
        "target_level": "senior",
        "gap_summary": "no number on the outcome",
    }

    [item] = spec.to_items([case], CTX)

    assert item.input["transcript"] == case["transcript"]
    assert item.input["grade"] == case["grade"]
    assert item.input["target_level"] == "senior"
    assert item.expected_output == {"gap_summary": case["gap_summary"]}


def _coach_report(summary: str) -> dict[str, Any]:
    return {
        "summary": summary,
        "answer_advice": [{"question": "q", "diagnosis": "d", "fix": "state the latency number"}],
        "drills": [{"focus": "results", "exercise": "rewrite with a metric"}],
        "study_plan": "practise quantified endings",
    }


def test_coach_evaluator_flags_house_voice_tells_in_the_prose() -> None:
    spec = dataset_specs()["coach"]

    clean_prose = _coach_report("Solid ownership, but the ending lacks a number.")
    [clean] = spec.evaluate(clean_prose, {}, {})
    assert clean.name == "coach_voice"
    assert clean.value == 1.0

    [slop] = spec.evaluate(_coach_report("Great answer! Truly impressive work."), {}, {})
    assert slop.value == 0.0
    assert slop.comment

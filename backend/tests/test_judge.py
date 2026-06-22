from typing import get_args, get_origin

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError
from stubs import StubChatModel

from sotellme.interviewer import Turn
from sotellme.judge import (
    CompetencyCoverage,
    CoverageVerdict,
    DimensionVerdict,
    JudgeError,
    QuestionContext,
    QuestionJudge,
    QuestionVerdict,
)


def _is_model(annotation: object) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _assert_flat(model: type[BaseModel]) -> None:
    """Assert no field nests a sibling model; a list of flat-item models is allowed."""
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if get_origin(annotation) in (list, tuple, set, frozenset):
            for item in get_args(annotation):
                if not _is_model(item):
                    continue
                nested = [n for n, f in item.model_fields.items() if _is_model(f.annotation)]
                assert not nested, f"{model.__name__}.{name}: {item.__name__} nests {nested}"
            continue
        assert not _is_model(annotation), f"{model.__name__}.{name} nests {annotation}"


def _dimension(score: int = 4) -> DimensionVerdict:
    return DimensionVerdict(rationale="Reasoned about it.", score=score)


def _verdict() -> QuestionVerdict:
    return QuestionVerdict(
        relevance_rationale="Reasoned about it.",
        relevance_score=4,
        probes_the_flagged_gap_rationale="Reasoned about it.",
        probes_the_flagged_gap_score=4,
        level_appropriateness_rationale="Reasoned about it.",
        level_appropriateness_score=4,
        non_leading_rationale="Reasoned about it.",
        non_leading_score=4,
        follow_up_discipline_rationale="Reasoned about it.",
        follow_up_discipline_score=4,
        overall_rationale="Solid probe.",
        overall="good",
    )


def _context(**overrides: object) -> QuestionContext:
    base = {
        "question": "You said the team cut latency - what did you personally change?",
        "competency": "ownership",
        "target_level": "senior",
        "gap": "blurred ownership: 'we cut latency' without his own part",
        "transcript": [Turn(question="Tell me about a project.", answer="We cut latency 38%.")],
        "sufficient_signal": False,
        "consecutive_follow_ups": 1,
    }
    base.update(overrides)
    return QuestionContext(**base)  # type: ignore[arg-type]


def _prompt_text(model: StubChatModel) -> str:
    return " ".join(text for _, text in model.seen_inputs[0])


def test_the_judge_returns_a_structured_question_verdict() -> None:
    model = StubChatModel(structured_response=_verdict())

    verdict = QuestionJudge(model).judge_question(_context())

    assert verdict == _verdict()


def test_the_judge_sees_the_question_context() -> None:
    model = StubChatModel(structured_response=_verdict())

    QuestionJudge(model).judge_question(_context())

    prompt = _prompt_text(model)
    assert "what did you personally change?" in prompt
    assert "ownership" in prompt
    assert "senior" in prompt
    assert "blurred ownership" in prompt
    assert "We cut latency 38%." in prompt


def test_the_judge_sees_the_follow_up_evidence_the_director_had() -> None:
    model = StubChatModel(structured_response=_verdict())

    QuestionJudge(model).judge_question(_context(sufficient_signal=True, consecutive_follow_ups=3))

    prompt = _prompt_text(model)
    assert "sufficient_signal=True" in prompt
    assert "consecutive_follow_ups=3" in prompt


def test_the_judge_prompt_carries_the_level_appropriateness_ladder() -> None:
    model = StubChatModel(structured_response=_verdict())

    QuestionJudge(model).judge_question(_context())

    prompt = _prompt_text(model)
    assert "Level-appropriateness ladder" in prompt
    assert "staff/principal" in prompt
    assert "undershooting" in prompt


def test_the_judge_retries_a_transient_question_structured_output_failure() -> None:
    model = StubChatModel(
        structured_response=_verdict(),
        structured_error=OutputParserException("dropped a score field"),
        structured_error_limit=1,
    )

    verdict = QuestionJudge(model).judge_question(_context())

    assert verdict == _verdict()
    assert model.structured_calls == 2


def test_the_judge_retries_a_transient_coverage_structured_output_failure() -> None:
    coverage = CoverageVerdict(
        competencies=[CompetencyCoverage(competency="ownership", status="covered")],
        rationale="solid",
        verdict="good",
    )
    model = StubChatModel(
        structured_response=coverage,
        structured_error=OutputParserException("dropped a field"),
        structured_error_limit=1,
    )

    verdict = QuestionJudge(model).judge_coverage("senior", [Turn(question="Q", answer="A")])

    assert verdict == coverage
    assert model.structured_calls == 2


def test_a_malformed_question_verdict_raises_judge_error() -> None:
    model = StubChatModel(
        structured_error=OutputParserException("bad"), structured_response=_verdict()
    )

    with pytest.raises(JudgeError):
        QuestionJudge(model).judge_question(_context())


def test_coverage_judge_returns_a_verdict_grading_the_level_emphasis() -> None:
    coverage = CoverageVerdict(
        competencies=[
            CompetencyCoverage(competency="strategic leadership", status="covered"),
            CompetencyCoverage(competency="developing others", status="missed"),
        ],
        rationale="Led on strategy but never touched mentoring.",
        verdict="weak",
    )
    model = StubChatModel(structured_response=coverage)
    transcript = [Turn(question="Q1", answer="A1")]

    verdict = QuestionJudge(model).judge_coverage("senior", transcript)

    assert verdict == coverage
    prompt = _prompt_text(model)
    assert "strategic leadership" in prompt
    assert "developing others" in prompt
    assert "A1" in prompt


def test_the_question_verdict_schema_stays_flat() -> None:
    _assert_flat(QuestionVerdict)


def test_the_coverage_verdict_schema_stays_flat() -> None:
    _assert_flat(CoverageVerdict)


def test_the_dimensions_property_reassembles_dimension_verdicts() -> None:
    dimensions = _verdict().dimensions

    assert set(dimensions) == {
        "relevance",
        "probes_the_flagged_gap",
        "level_appropriateness",
        "non_leading",
        "follow_up_discipline",
    }
    assert dimensions["relevance"] == _dimension()


def _raw_question_args() -> dict[str, object]:
    return {
        "relevance_rationale": "targets the ownership competency in play",
        "relevance_score": 4,
        "probes_the_flagged_gap_rationale": "chases the blurred-ownership gap directly",
        "probes_the_flagged_gap_score": 5,
        "level_appropriateness_rationale": "asks for personal contribution at senior depth",
        "level_appropriateness_score": 4,
        "non_leading_rationale": "open-ended, hands over nothing",
        "non_leading_score": 5,
        "follow_up_discipline_rationale": "right to probe again given thin signal",
        "follow_up_discipline_score": 4,
        "overall_rationale": "a sharp, well-targeted probe",
        "overall": "good",
    }


def test_a_question_verdict_validates_from_raw_tool_call_args() -> None:
    verdict = QuestionVerdict.model_validate(_raw_question_args())

    assert verdict.overall == "good"
    assert verdict.dimensions["probes_the_flagged_gap"].score == 5


def test_the_collapsed_nested_payload_opus_emitted_is_rejected() -> None:
    # The shape claude-opus-4-8 produced live: all five dimensions collapsed into one
    # string field (with leaked tool-call XML) plus a single score, the rest missing.
    payload = {
        "relevance": '<parameter name="rationale">targets ownership</parameter>',
        "score": 5,
        "overall_rationale": "a sharp probe",
        "overall": "good",
    }

    with pytest.raises(ValidationError):
        QuestionVerdict.model_validate(payload)


def test_a_judge_error_diagnostic_surfaces_the_underlying_validation_cause() -> None:
    try:
        QuestionVerdict.model_validate({"overall": "good"})
    except ValidationError as cause:
        exc = JudgeError("Could not judge the question.")
        exc.__cause__ = cause

    detail = exc.diagnostic()

    assert "Could not judge the question." in detail
    assert "ValidationError" in detail
    assert "relevance_rationale" in detail


def test_a_judge_error_diagnostic_without_a_cause_returns_the_message() -> None:
    assert JudgeError("boom").diagnostic() == "boom"


def test_a_coverage_verdict_validates_from_raw_tool_call_args() -> None:
    payload = {
        "competencies": [
            {"competency": "strategic leadership", "status": "covered"},
            {"competency": "developing others", "status": "missed"},
        ],
        "rationale": "led on strategy but never touched mentoring",
        "verdict": "weak",
    }

    verdict = CoverageVerdict.model_validate(payload)

    assert verdict.verdict == "weak"
    assert verdict.competencies[1].status == "missed"

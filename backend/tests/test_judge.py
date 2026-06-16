import pytest
from langchain_core.exceptions import OutputParserException
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


def _dimension(score: int = 4) -> DimensionVerdict:
    return DimensionVerdict(rationale="Reasoned about it.", score=score)


def _verdict() -> QuestionVerdict:
    return QuestionVerdict(
        relevance=_dimension(),
        probes_the_flagged_gap=_dimension(),
        level_appropriateness=_dimension(),
        non_leading=_dimension(),
        follow_up_discipline=_dimension(),
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

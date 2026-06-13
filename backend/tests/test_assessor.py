import pytest
from pydantic import ValidationError
from stubs import StubChatModel

from sotellme.assessor import AnswerAssessment, AssessorError, StarFlags, assess_answer
from sotellme.interviewer import Turn

TRANSCRIPT = [
    Turn(
        question="Tell me about the migration at Acme.",
        answer="I led the move to Kafka and latency dropped from 4 hours to 90 seconds.",
    )
]

ASSESSMENT = AnswerAssessment(
    star=StarFlags(situation=False, task=False, action=True, result=True, quantified_result=True),
    sufficient_signal=False,
    claims_worth_chasing=["latency dropped from 4 hours to 90 seconds"],
)


def test_an_answer_is_assessed_into_a_structured_assessment() -> None:
    model = StubChatModel(structured_response=ASSESSMENT)

    assessment = assess_answer("the Acme migration", TRANSCRIPT, model)

    assert assessment == ASSESSMENT


def test_the_assessor_sees_the_topic_and_the_transcript() -> None:
    model = StubChatModel(structured_response=ASSESSMENT)

    assess_answer("the Acme migration", TRANSCRIPT, model)

    human_text = " ".join(text for _, text in model.seen_inputs[0])
    assert "the Acme migration" in human_text
    assert "I led the move to Kafka" in human_text


def test_a_failed_parse_is_a_clear_error() -> None:
    error = ValidationError.from_exception_data("AnswerAssessment", [])
    model = StubChatModel(structured_error=error)

    with pytest.raises(AssessorError, match="Could not assess"):
        assess_answer("the Acme migration", TRANSCRIPT, model)


def test_a_non_assessment_response_is_a_clear_error() -> None:
    model = StubChatModel(structured_response={"not": "an assessment"})

    with pytest.raises(AssessorError, match="Could not assess"):
        assess_answer("the Acme migration", TRANSCRIPT, model)

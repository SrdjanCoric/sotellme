from collections.abc import Sequence

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

from sotellme.caching import cache_system_prompt
from sotellme.interviewer import Turn, render_transcript
from sotellme.prompts import assessor_messages


class StarFlags(BaseModel):
    situation: bool = Field(
        description="The answer sets the scene: the team, company, or context the story happens in."
    )
    task: bool = Field(
        description="The answer names a concrete problem or goal the candidate had to address."
    )
    action: bool = Field(description="The answer describes what the candidate actually did.")
    result: bool = Field(description="The answer states an outcome: how things ended up.")
    quantified_result: bool = Field(
        description="The stated outcome carries a number or other measurable change."
    )


class AnswerAssessment(BaseModel):
    star: StarFlags = Field(
        description="Which story elements the latest answer states, as evidence."
    )
    sufficient_signal: bool = Field(
        description=(
            "True when the conversation on the current topic now holds enough concrete "
            "evidence of how the candidate works that more questions on it would add little."
        )
    )
    claims_worth_chasing: list[str] = Field(
        description=(
            "Claims from the latest answer worth a follow-up, quoted near-verbatim and "
            "ordered most interesting first. Empty when nothing stands out."
        )
    )


class TopicAssessment(BaseModel):
    topic: str
    assessment: AnswerAssessment


class AssessorError(Exception):
    pass


_ASSESS_FAILURE_MESSAGE = "Could not assess the answer. Try answering again."


def assess_answer(
    topic: str, transcript: Sequence[Turn], model: BaseChatModel, provider: str = ""
) -> AnswerAssessment:
    structured = model.with_structured_output(AnswerAssessment)
    try:
        messages = cache_system_prompt(
            assessor_messages(topic, render_transcript(transcript)), provider
        )
        result = structured.invoke(messages)
    except (ValidationError, OutputParserException) as exc:
        raise AssessorError(_ASSESS_FAILURE_MESSAGE) from exc
    if not isinstance(result, AnswerAssessment):
        raise AssessorError(_ASSESS_FAILURE_MESSAGE)
    return result

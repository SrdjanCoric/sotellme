from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from sotellme.coverage import StarFlags
from sotellme.prompts import star_flagger_messages


class StarFlaggerError(Exception):
    pass


_FLAG_FAILURE_MESSAGE = "Could not read the answer's story elements. Try answering again."


def flag_star_elements(answer: str, model: BaseChatModel) -> StarFlags:
    structured = model.with_structured_output(StarFlags)
    try:
        result = structured.invoke(star_flagger_messages(answer))
    except (ValidationError, OutputParserException) as exc:
        raise StarFlaggerError(_FLAG_FAILURE_MESSAGE) from exc
    if not isinstance(result, StarFlags):
        raise StarFlaggerError(_FLAG_FAILURE_MESSAGE)
    return result

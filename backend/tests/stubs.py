from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ConfigDict, Field


class StubChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    structured_response: Any = None
    structured_error: Exception | None = None
    text_response: str = ""
    seen_inputs: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_inputs.append([(m.type, m.text) for m in messages])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(self.text_response))])

    def with_structured_output(
        self, schema: dict[str, Any] | type, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        def run(value: LanguageModelInput) -> dict[str, Any] | BaseModel:
            self.seen_inputs.append(value)
            if self.structured_error is not None:
                raise self.structured_error
            result: dict[str, Any] | BaseModel = self.structured_response
            return result

        return RunnableLambda(run)

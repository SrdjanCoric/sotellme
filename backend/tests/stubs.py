from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


class ToolLoopStubModel(BaseChatModel):
    """Replays scripted AI messages; repeats the last one when the script runs out."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: list[AIMessage] = Field(default_factory=list)
    seen_message_lists: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)
    replies_given: int = 0

    @property
    def _llm_type(self) -> str:
        return "tool-loop-stub"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.bound_tools = list(tools)
        return cast(Runnable[LanguageModelInput, AIMessage], self)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_message_lists.append(list(messages))
        index = min(self.replies_given, len(self.script) - 1)
        self.replies_given += 1
        return ChatResult(generations=[ChatGeneration(message=self.script[index])])


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

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

from sotellme.interviewer import render_role_context
from sotellme.prompts import RESEARCH_WRAP_INSTRUCTION, research_messages
from sotellme.role import RoleContext

DEFAULT_FETCH_CAP = 6

FETCH_BUDGET_MESSAGE = (
    "The fetch budget for this session is used up. Write the brief from what you already have."
)

PageFetcher = Callable[[str], str]


class fetch_page(BaseModel):
    """Fetch one public web page and return its visible text."""

    url: str = Field(description="The full http or https address of the page to fetch.")


def build_company_brief(
    posting_text: str,
    context: RoleContext,
    model: BaseChatModel,
    fetcher: PageFetcher,
    max_fetches: int = DEFAULT_FETCH_CAP,
) -> str:
    prompt = dict(research_messages(render_role_context(context), posting_text, max_fetches))
    messages: list[BaseMessage] = [SystemMessage(prompt["system"]), HumanMessage(prompt["human"])]
    bound = model.bind_tools([fetch_page])
    fetches_used = 0
    for _ in range(max_fetches + 2):
        response = bound.invoke(messages)
        messages.append(response)
        if not isinstance(response, AIMessage) or not response.tool_calls:
            return response.text.strip()
        for call in response.tool_calls:
            if fetches_used >= max_fetches:
                content = FETCH_BUDGET_MESSAGE
            else:
                fetches_used += 1
                try:
                    content = fetcher(str(call["args"].get("url", "")))
                except Exception as exc:  # a failed fetch must never kill the session
                    content = f"Could not fetch the page: {exc}"
            messages.append(ToolMessage(content=content, tool_call_id=call.get("id") or ""))
    messages.append(HumanMessage(RESEARCH_WRAP_INSTRUCTION))
    return model.invoke(messages).text.strip()

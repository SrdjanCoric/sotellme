"""Agentic company research that drafts a brief from a posting and fetched web pages."""

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
    """Research a company and return a written brief built by the model.

    The model is given the role context and posting and may call the ``fetch_page`` tool to
    read web pages, up to ``max_fetches`` times; further fetch requests are answered with a
    budget-used-up message. A failed fetch is reported back to the model rather than raised.
    The loop runs for a bounded number of rounds, and a final wrap-up instruction forces a
    brief if the model is still requesting tools at the end.

    Args:
        posting_text: The job posting text to research from.
        context: The role context describing the candidate and target role.
        model: Chat model used to drive the research and write the brief.
        fetcher: Callable that fetches a URL and returns its visible text.
        max_fetches: Maximum number of page fetches allowed for the session.

    Returns:
        The company brief text, stripped of surrounding whitespace.
    """
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

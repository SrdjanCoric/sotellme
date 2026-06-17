from typing import Any, cast

from langchain_core.messages import AIMessage, ToolMessage
from stubs import ToolLoopStubModel

from sotellme.research import DEFAULT_FETCH_CAP, FETCH_BUDGET_MESSAGE, build_company_brief
from sotellme.role import CompetencyWeight, RoleContext

CONTEXT = RoleContext(
    company="Acme",
    role_title="Backend Engineer",
    competencies=[CompetencyWeight(name="ownership", weight=5)],
)

POSTING = "Backend Engineer at Acme. We build billing software for veterinary clinics."

BRIEF = "Acme makes billing software for veterinary clinics, sold to clinic chains."


def fetch_request(url: str, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "fetch_page", "args": {"url": url}, "id": call_id}],
    )


class RecordingFetcher:
    def __init__(self, page_text: str = "Acme product page.") -> None:
        self.page_text = page_text
        self.fetched_urls: list[str] = []

    def __call__(self, url: str) -> str:
        self.fetched_urls.append(url)
        return self.page_text


def test_the_agent_fetches_a_page_and_writes_the_brief() -> None:
    model = ToolLoopStubModel(script=[fetch_request("https://acme.com"), AIMessage(BRIEF)])
    fetcher = RecordingFetcher()

    brief = build_company_brief(POSTING, CONTEXT, model, fetcher)

    assert brief == BRIEF
    assert fetcher.fetched_urls == ["https://acme.com"]
    tool_messages = [
        message for message in model.seen_message_lists[-1] if isinstance(message, ToolMessage)
    ]
    assert tool_messages and "Acme product page." in str(tool_messages[0].content)


def test_the_agent_sees_the_posting_and_role_details() -> None:
    model = ToolLoopStubModel(script=[AIMessage(BRIEF)])

    build_company_brief(POSTING, CONTEXT, model, RecordingFetcher())

    seen = " ".join(message.text for message in model.seen_message_lists[0])
    assert "billing software for veterinary clinics" in seen
    assert "Company: Acme" in seen


def test_fetches_never_exceed_the_cap() -> None:
    model = ToolLoopStubModel(script=[fetch_request("https://acme.com")])
    fetcher = RecordingFetcher()

    build_company_brief(POSTING, CONTEXT, model, fetcher, max_fetches=3)

    assert len(fetcher.fetched_urls) == 3
    budget_refusals = [
        message
        for message in model.seen_message_lists[-1]
        if isinstance(message, ToolMessage) and FETCH_BUDGET_MESSAGE in str(message.content)
    ]
    assert budget_refusals


def test_the_default_cap_sits_in_the_decided_range() -> None:
    assert 5 <= DEFAULT_FETCH_CAP <= 8


def test_a_failed_fetch_becomes_tool_feedback_not_a_crash() -> None:
    def failing_fetcher(url: str) -> str:
        raise RuntimeError("connection reset")

    model = ToolLoopStubModel(script=[fetch_request("https://acme.com"), AIMessage(BRIEF)])

    brief = build_company_brief(POSTING, CONTEXT, model, failing_fetcher)

    assert brief == BRIEF
    failures = [
        message
        for message in model.seen_message_lists[-1]
        if isinstance(message, ToolMessage) and "connection reset" in str(message.content)
    ]
    assert failures


def test_a_tool_call_without_an_id_does_not_crash() -> None:
    request = fetch_request("https://acme.com")
    cast(dict[str, Any], request.tool_calls[0]).pop("id", None)
    model = ToolLoopStubModel(script=[request, AIMessage(BRIEF)])

    brief = build_company_brief(POSTING, CONTEXT, model, RecordingFetcher())

    assert brief == BRIEF


def test_fetched_pages_are_framed_as_untrusted_data() -> None:
    model = ToolLoopStubModel(script=[AIMessage(BRIEF)])

    build_company_brief(POSTING, CONTEXT, model, RecordingFetcher())

    system_text = model.seen_message_lists[0][0].text.lower()
    assert "data" in system_text
    assert "not instructions" in system_text

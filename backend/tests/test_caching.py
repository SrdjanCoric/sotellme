import pytest
from langchain_core.messages import SystemMessage
from stubs import StubChatModel

from sotellme.caching import MIN_CACHEABLE_CHARS, cache_system_prompt
from sotellme.grader import SessionGrade, grade_session
from sotellme.interviewer import Turn
from sotellme.prompts import (
    GRADER_SYSTEM_PROMPT,
    assessor_messages,
    coach_messages,
    director_messages,
    grader_messages,
    guardrail_messages,
    profile_extraction_messages,
    question_messages,
    research_messages,
    role_context_messages,
)

LONG_SYSTEM = "You are an agent with a long, stable system prompt. " * 100


def system_block(message: object) -> dict[str, object]:
    assert isinstance(message, SystemMessage)
    assert isinstance(message.content, list)
    block = message.content[0]
    assert isinstance(block, dict)
    return block


def _director_case(transcript: str) -> list[tuple[str, str]]:
    return director_messages(
        role_details="Company: Acme",
        emphasis=("ownership",),
        brief="Acme builds billing software.",
        profile_text="Engineer at Acme",
        transcript_text=transcript,
        assessment_notes="No answers assessed yet.",
        questions_asked=1,
        question_cap=20,
    )


# Each agent built twice with different per-case content (CV, posting, transcript, level, ...).
AGENT_CASES = [
    (
        "parser",
        profile_extraction_messages("CV one"),
        profile_extraction_messages("a different and longer CV body"),
    ),
    (
        "role_builder",
        role_context_messages("posting one"),
        role_context_messages("an entirely different posting"),
    ),
    (
        "researcher",
        research_messages("role A", "posting one", 3),
        research_messages("role B", "posting two", 5),
    ),
    (
        "assessor",
        assessor_messages("topic A", "Q: a\nA: b"),
        assessor_messages("topic B", "Q: a\nA: b\nQ: c\nA: d"),
    ),
    (
        "interviewer",
        question_messages("role A", "brief A", "profile A", "Q: a\nA: b", "directive A"),
        question_messages(
            "role B", "brief B", "profile B", "Q: a\nA: b\nQ: c\nA: d", "directive B"
        ),
    ),
    (
        "guardrail",
        guardrail_messages("question A", "answer A"),
        guardrail_messages("question B", "a longer different answer"),
    ),
    (
        "grader",
        grader_messages("senior", "Q: a\nA: b"),
        grader_messages("junior", "Q: a\nA: b\nQ: c\nA: d"),
    ),
    (
        "coach",
        coach_messages("senior", "Q: a\nA: b", "grade A"),
        coach_messages("junior", "Q: a\nA: b\nQ: c\nA: d", "grade B"),
    ),
    ("director", _director_case("Q: a\nA: b"), _director_case("Q: a\nA: b\nQ: c\nA: d")),
]


@pytest.mark.parametrize("agent,case_a,case_b", AGENT_CASES, ids=[case[0] for case in AGENT_CASES])
def test_each_agents_system_prompt_is_a_stable_leading_block_across_cases(
    agent: str,
    case_a: list[tuple[str, str]],
    case_b: list[tuple[str, str]],
) -> None:
    assert case_a[0][0] == "system"
    assert case_a[0] == case_b[0], f"{agent} leaks per-case data into its system prompt"


def test_the_anthropic_cache_block_is_identical_across_cases() -> None:
    case_a = cache_system_prompt(grader_messages("senior", "Q: a\nA: b"), "anthropic")
    case_b = cache_system_prompt(grader_messages("junior", "Q: x\nA: y\nQ: z\nA: w"), "anthropic")

    assert system_block(case_a[0]) == system_block(case_b[0])


def test_per_case_content_is_left_uncached() -> None:
    cached = cache_system_prompt(grader_messages("senior", "Q: a\nA: b"), "anthropic")

    human = cached[1]
    assert isinstance(human, tuple)
    assert human[0] == "human"


def test_an_anthropic_agent_sends_a_cache_controlled_system_prompt() -> None:
    stub = StubChatModel(structured_response=SessionGrade(scores=[]))

    grade_session([Turn(question="q", answer="a")], "senior", stub, provider="anthropic")

    sent = stub.seen_inputs[-1]
    assert system_block(sent[0])["cache_control"] == {"type": "ephemeral"}


def test_an_agent_without_a_provider_sends_a_plain_system_prompt() -> None:
    stub = StubChatModel(structured_response=SessionGrade(scores=[]))

    grade_session([Turn(question="q", answer="a")], "senior", stub)

    sent = stub.seen_inputs[-1]
    assert sent[0] == ("system", GRADER_SYSTEM_PROMPT)


def test_anthropic_marks_the_system_prompt_with_an_ephemeral_cache_breakpoint() -> None:
    messages = [("system", LONG_SYSTEM), ("human", "case-specific content")]

    cached = cache_system_prompt(messages, "anthropic")

    block = system_block(cached[0])
    assert block["text"] == LONG_SYSTEM
    assert block["cache_control"] == {"type": "ephemeral"}


def test_auto_caching_providers_get_the_messages_unchanged() -> None:
    messages = [("system", LONG_SYSTEM), ("human", "case-specific content")]

    for provider in ("openai", "google_genai"):
        assert cache_system_prompt(messages, provider) == messages


def test_a_system_prompt_below_the_minimum_is_not_marked_for_anthropic() -> None:
    tiny = "x" * (MIN_CACHEABLE_CHARS - 1)
    messages = [("system", tiny), ("human", "case-specific content")]

    assert cache_system_prompt(messages, "anthropic") == messages

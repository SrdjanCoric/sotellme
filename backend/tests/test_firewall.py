import re

from sotellme.coverage import StarFlags
from sotellme.prompts import (
    CLOSING_HUMAN_TEMPLATE,
    CLOSING_SYSTEM_PROMPT,
    GAP_GUIDANCE,
    HOUSE_VOICE,
    INTERVIEWER_SYSTEM_PROMPT,
    OPENING_QUESTION_HUMAN_TEMPLATE,
    PROBE_QUESTION_HUMAN_TEMPLATE,
    STAR_FLAGGER_HUMAN_TEMPLATE,
    STAR_FLAGGER_SYSTEM_PROMPT,
    closing_messages,
    opening_question_messages,
    probe_question_messages,
    star_flagger_messages,
)

RUBRIC_MARKERS = (
    "rubric",
    "score",
    "scoring",
    "grade",
    "grading",
    "four dimensions",
    "three keys",
    "leveling",
    "junior",
    "mid-level",
    "senior",
    "staff",
    "target level",
)

NEUTRAL_PROFILE = "Engineer at Acme\n- Cut latency by 38%"
NEUTRAL_TRANSCRIPT = "Q: Tell me about the latency work.\nA: We migrated the pipeline."


def rubric_markers_in(text: str) -> list[str]:
    return [
        marker
        for marker in RUBRIC_MARKERS
        if re.search(rf"\b{re.escape(marker)}\b", text, re.IGNORECASE)
    ]


def interviewer_and_flagger_prompt_assembly() -> list[str]:
    artifacts = [
        HOUSE_VOICE,
        INTERVIEWER_SYSTEM_PROMPT,
        OPENING_QUESTION_HUMAN_TEMPLATE,
        PROBE_QUESTION_HUMAN_TEMPLATE,
        CLOSING_SYSTEM_PROMPT,
        CLOSING_HUMAN_TEMPLATE,
        STAR_FLAGGER_SYSTEM_PROMPT,
        STAR_FLAGGER_HUMAN_TEMPLATE,
        *GAP_GUIDANCE.values(),
    ]
    for messages in (
        opening_question_messages(NEUTRAL_PROFILE),
        probe_question_messages(NEUTRAL_PROFILE, NEUTRAL_TRANSCRIPT, ("result",)),
        closing_messages(NEUTRAL_TRANSCRIPT),
        star_flagger_messages("We migrated the pipeline."),
    ):
        artifacts.extend(text for _, text in messages)
    artifacts.extend(field.description or "" for field in StarFlags.model_fields.values())
    return artifacts


def test_the_scanner_itself_detects_rubric_language() -> None:
    leaked = "Score this answer against the rubric and the Four Dimensions for a Senior."

    assert set(rubric_markers_in(leaked)) == {"score", "rubric", "four dimensions", "senior"}


def test_nothing_rubric_derived_enters_interviewer_or_flagger_prompts() -> None:
    leaks = {
        text: markers
        for text in interviewer_and_flagger_prompt_assembly()
        if (markers := rubric_markers_in(text))
    }

    assert not leaks, f"rubric language leaked into interviewer/flagger prompts: {leaks}"

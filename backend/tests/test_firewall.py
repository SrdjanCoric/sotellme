import re

from sotellme.assessor import AnswerAssessment, StarFlags
from sotellme.director import DirectorDecision
from sotellme.prompts import (
    ASSESSOR_HUMAN_TEMPLATE,
    ASSESSOR_SYSTEM_PROMPT,
    CLOSING_HUMAN_TEMPLATE,
    CLOSING_SYSTEM_PROMPT,
    DIRECTOR_HUMAN_TEMPLATE,
    DIRECTOR_SYSTEM_PROMPT,
    FOLLOW_UP_DIRECTIVE_TEMPLATE,
    HOUSE_VOICE,
    INTERVIEWER_SYSTEM_PROMPT,
    NEW_TOPIC_DIRECTIVE_TEMPLATE,
    QUESTION_HUMAN_TEMPLATE,
    assessor_messages,
    closing_messages,
    director_messages,
    question_messages,
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
NEUTRAL_BRIEF = "Acme builds billing software for veterinary clinics."


def rubric_markers_in(text: str) -> list[str]:
    return [
        marker
        for marker in RUBRIC_MARKERS
        if re.search(rf"\b{re.escape(marker)}\b", text, re.IGNORECASE)
    ]


def candidate_facing_and_mid_session_prompt_assembly() -> list[str]:
    artifacts = [
        HOUSE_VOICE,
        INTERVIEWER_SYSTEM_PROMPT,
        QUESTION_HUMAN_TEMPLATE,
        FOLLOW_UP_DIRECTIVE_TEMPLATE,
        NEW_TOPIC_DIRECTIVE_TEMPLATE,
        CLOSING_SYSTEM_PROMPT,
        CLOSING_HUMAN_TEMPLATE,
        ASSESSOR_SYSTEM_PROMPT,
        ASSESSOR_HUMAN_TEMPLATE,
        DIRECTOR_SYSTEM_PROMPT,
        DIRECTOR_HUMAN_TEMPLATE,
    ]
    for messages in (
        question_messages(
            "Company: Acme",
            NEUTRAL_BRIEF,
            NEUTRAL_PROFILE,
            NEUTRAL_TRANSCRIPT,
            "The interview now turns to: their background. Ask one question that opens it.",
        ),
        closing_messages(NEUTRAL_TRANSCRIPT),
        assessor_messages("their background", NEUTRAL_TRANSCRIPT),
        director_messages(
            role_details="Company: Acme",
            emphasis=("problem solving",),
            brief=NEUTRAL_BRIEF,
            profile_text=NEUTRAL_PROFILE,
            transcript_text=NEUTRAL_TRANSCRIPT,
            assessment_notes=(
                "After answer 1 (topic: their background): the topic needs more signal"
            ),
            questions_asked=1,
            question_cap=20,
        ),
    ):
        artifacts.extend(text for _, text in messages)
    for model in (StarFlags, AnswerAssessment, DirectorDecision):
        artifacts.extend(field.description or "" for field in model.model_fields.values())
    return artifacts


def test_the_scanner_itself_detects_rubric_language() -> None:
    leaked = "Score this answer against the rubric and the Four Dimensions for a Senior."

    assert set(rubric_markers_in(leaked)) == {"score", "rubric", "four dimensions", "senior"}


def test_nothing_rubric_derived_enters_non_grader_prompts() -> None:
    leaks = {
        text: markers
        for text in candidate_facing_and_mid_session_prompt_assembly()
        if (markers := rubric_markers_in(text))
    }

    assert not leaks, f"rubric language leaked into interviewer/assessor/director prompts: {leaks}"


def test_the_assessors_judgment_never_enters_candidate_facing_prompts() -> None:
    question = question_messages(
        "Company: Acme",
        NEUTRAL_BRIEF,
        NEUTRAL_PROFILE,
        NEUTRAL_TRANSCRIPT,
        "Follow up on this from their last answer: the migration claim.",
    )
    closing = closing_messages(NEUTRAL_TRANSCRIPT)

    for _, text in (*question, *closing):
        lowered = text.lower()
        assert "assessment notes" not in lowered
        assert "sufficient" not in lowered
        assert "enough signal" not in lowered

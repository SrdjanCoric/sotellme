from stubs import StubChatModel

from sotellme.interviewer import (
    LLMInterviewer,
    Turn,
    render_profile,
    render_role_context,
    render_transcript,
)
from sotellme.profile import CandidateProfile, Project, Role
from sotellme.prompts import GAP_GUIDANCE, MOTIVATION_GUIDANCE
from sotellme.role import CompetencyWeight, RoleContext

PROFILE = CandidateProfile(
    roles=[Role(title="Software Engineer", organization="Acme", period="2020-2024")],
    projects=[Project(name="openroster", description="Shift-planning library")],
    quantified_claims=["Cut latency by 38%"],
    technologies=["Python", "Kafka"],
)


def test_profile_renders_every_field_for_the_prompt() -> None:
    text = render_profile(PROFILE)

    assert "Software Engineer, Acme (2020-2024)" in text
    assert "openroster: Shift-planning library" in text
    assert "Cut latency by 38%" in text
    assert "Python, Kafka" in text


def test_transcript_renders_as_question_answer_turns() -> None:
    transcript = [Turn(question="What happened?", answer="We migrated.")]

    assert render_transcript(transcript) == "Q: What happened?\nA: We migrated."


def test_role_context_renders_without_the_target_level() -> None:
    context = RoleContext(
        company="Acme",
        role_title="Backend Engineer",
        competencies=[CompetencyWeight(name="ownership", weight=5)],
        framework="Acme Operating Principles",
        target_level="senior",
    )

    text = render_role_context(context)

    assert "Company: Acme" in text
    assert "Role: Backend Engineer" in text
    assert "Acme Operating Principles" in text
    assert "senior" not in text.lower()


def test_competency_question_is_grounded_in_the_rendered_profile() -> None:
    model = StubChatModel(text_response="  Tell me about the latency work at Acme?  ")
    interviewer = LLMInterviewer(model)

    question = interviewer.competency_question(PROFILE, [], "impact")

    assert question == "Tell me about the latency work at Acme?"
    human_texts = [text for role, text in model.seen_inputs[0] if role == "human"]
    assert "Cut latency by 38%" in human_texts[0]
    assert "impact" in human_texts[0]


def test_motivation_question_carries_the_posting_and_topic() -> None:
    model = StubChatModel(text_response="Why Acme, of all places?")
    interviewer = LLMInterviewer(model)
    context = RoleContext(
        company="Acme",
        competencies=[CompetencyWeight(name="ownership", weight=5)],
    )
    transcript = [Turn(question="What happened?", answer="We migrated.")]

    question = interviewer.motivation_question(
        context, "Acme builds billing software.", transcript, "company"
    )

    assert question == "Why Acme, of all places?"
    human_texts = [text for role, text in model.seen_inputs[0] if role == "human"]
    assert "Acme builds billing software." in human_texts[0]
    assert MOTIVATION_GUIDANCE["company"] in human_texts[0]


def test_probe_question_carries_the_transcript_and_the_primary_gap() -> None:
    model = StubChatModel(text_response="And how did that end up?")
    interviewer = LLMInterviewer(model)
    transcript = [Turn(question="What happened?", answer="We migrated.")]

    question = interviewer.probe_question(PROFILE, transcript, ("result", "quantified_result"))

    assert question == "And how did that end up?"
    human_texts = [text for role, text in model.seen_inputs[0] if role == "human"]
    assert "Q: What happened?\nA: We migrated." in human_texts[0]
    assert GAP_GUIDANCE["result"] in human_texts[0]


def test_interviewer_output_is_sanitized_of_ai_dashes() -> None:
    model = StubChatModel(text_response="The pipeline shipped—what changed after that?")
    interviewer = LLMInterviewer(model)
    transcript = [Turn(question="What happened?", answer="We migrated.")]

    question = interviewer.probe_question(PROFILE, transcript, ("result",))

    assert question == "The pipeline shipped - what changed after that?"


def test_closing_turn_carries_the_transcript() -> None:
    model = StubChatModel(text_response="  That covers it, thanks.  ")
    interviewer = LLMInterviewer(model)
    transcript = [Turn(question="What happened?", answer="We migrated.")]

    closing = interviewer.closing_turn(transcript)

    assert closing == "That covers it, thanks."
    human_texts = [text for role, text in model.seen_inputs[0] if role == "human"]
    assert "Q: What happened?\nA: We migrated." in human_texts[0]

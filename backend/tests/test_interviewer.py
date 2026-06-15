from stubs import StubChatModel

from sotellme.director import DirectorDecision
from sotellme.interviewer import (
    LLMInterviewer,
    Turn,
    render_directive,
    render_profile,
    render_role_context,
    render_transcript,
)
from sotellme.profile import CandidateProfile, Project, Role
from sotellme.role import CompetencyWeight, RoleContext

PROFILE = CandidateProfile(
    roles=[Role(title="Software Engineer", organization="Acme", period="2020-2024")],
    projects=[Project(name="openroster", description="Shift-planning library")],
    quantified_claims=["Cut latency by 38%"],
    technologies=["Python", "Kafka"],
)

CONTEXT = RoleContext(
    company="Acme",
    role_title="Backend Engineer",
    competencies=[CompetencyWeight(name="ownership", weight=5)],
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


def test_a_follow_up_directive_names_the_subject_and_reason() -> None:
    decision = DirectorDecision(
        action="follow_up", subject="the migration claim", reason="impact left unexplained"
    )

    directive = render_directive(decision)

    assert "Follow up on this from their last answer: the migration claim." in directive
    assert "impact left unexplained" in directive


def test_a_new_topic_directive_names_the_topic() -> None:
    decision = DirectorDecision(
        action="new_topic", subject="their most significant project", reason="the deep dive"
    )

    directive = render_directive(decision)

    assert "The interview now turns to: their most significant project." in directive
    assert "the deep dive" in directive


def test_a_follow_up_directive_becomes_a_question_with_full_grounding() -> None:
    model = StubChatModel(text_response="  How did the scheduler rewrite land?  ")
    interviewer = LLMInterviewer(model)
    transcript = [Turn(question="What happened?", answer="I rewrote the scheduler alone.")]
    decision = DirectorDecision(
        action="follow_up", subject="rewrote the scheduler alone", reason="ownership signal"
    )

    question = interviewer.question_for(
        decision, PROFILE, CONTEXT, "Acme builds billing software.", transcript
    )

    assert question == "How did the scheduler rewrite land?"
    human_texts = [text for role, text in model.seen_inputs[0] if role == "human"]
    assert "rewrote the scheduler alone" in human_texts[0]
    assert "ownership signal" in human_texts[0]
    assert "Acme builds billing software." in human_texts[0]
    assert "Cut latency by 38%" in human_texts[0]
    assert "Q: What happened?" in human_texts[0]


def test_a_new_topic_directive_on_an_empty_transcript_opens_the_interview() -> None:
    model = StubChatModel(text_response="So, tell me a bit about yourself?")
    interviewer = LLMInterviewer(model)
    decision = DirectorDecision(
        action="new_topic", subject="who they are and their background", reason="the opener"
    )

    question = interviewer.question_for(decision, PROFILE, CONTEXT, "", [])

    assert question == "So, tell me a bit about yourself?"
    human_texts = [text for role, text in model.seen_inputs[0] if role == "human"]
    assert "who they are and their background" in human_texts[0]
    assert "<transcript>" not in human_texts[0]
    assert "<brief>" not in human_texts[0]


def test_interviewer_output_is_sanitized_of_ai_dashes() -> None:
    model = StubChatModel(text_response="The pipeline shipped—what changed after that?")
    interviewer = LLMInterviewer(model)
    transcript = [Turn(question="What happened?", answer="We migrated.")]
    decision = DirectorDecision(action="follow_up", subject="the pipeline", reason="the ending")

    question = interviewer.question_for(decision, PROFILE, CONTEXT, "", transcript)

    assert question == "The pipeline shipped - what changed after that?"


def test_closing_turn_carries_the_transcript() -> None:
    model = StubChatModel(text_response="  That covers it, thanks.  ")
    interviewer = LLMInterviewer(model)
    transcript = [Turn(question="What happened?", answer="We migrated.")]

    closing = interviewer.closing_turn(transcript)

    assert closing == "That covers it, thanks."
    human_texts = [text for role, text in model.seen_inputs[0] if role == "human"]
    assert "Q: What happened?\nA: We migrated." in human_texts[0]


def test_redirect_turn_points_back_to_the_question_and_is_sanitized() -> None:
    model = StubChatModel(text_response="  Let's stay with the interview—back to my question.  ")
    interviewer = LLMInterviewer(model)

    redirect = interviewer.redirect_turn("What problem was openroster solving?")

    assert redirect == "Let's stay with the interview - back to my question."
    human_texts = [text for role, text in model.seen_inputs[0] if role == "human"]
    assert "What problem was openroster solving?" in human_texts[0]

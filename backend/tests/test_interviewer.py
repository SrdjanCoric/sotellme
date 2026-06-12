from stubs import StubChatModel

from sotellme.interviewer import LLMInterviewer, Turn, render_profile, render_transcript
from sotellme.profile import CandidateProfile, Project, Role
from sotellme.prompts import GAP_GUIDANCE

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


def test_opening_question_is_grounded_in_the_rendered_profile() -> None:
    model = StubChatModel(text_response="  Tell me about the latency work at Acme?  ")
    interviewer = LLMInterviewer(model)

    question = interviewer.opening_question(PROFILE)

    assert question == "Tell me about the latency work at Acme?"
    human_texts = [text for role, text in model.seen_inputs[0] if role == "human"]
    assert "Cut latency by 38%" in human_texts[0]


def test_probe_question_carries_the_transcript_and_the_primary_gap() -> None:
    model = StubChatModel(text_response="And how did that end up?")
    interviewer = LLMInterviewer(model)
    transcript = [Turn(question="What happened?", answer="We migrated.")]

    question = interviewer.probe_question(PROFILE, transcript, ("result", "quantified_result"))

    assert question == "And how did that end up?"
    human_texts = [text for role, text in model.seen_inputs[0] if role == "human"]
    assert "Q: What happened?\nA: We migrated." in human_texts[0]
    assert GAP_GUIDANCE["result"] in human_texts[0]


def test_closing_turn_carries_the_transcript() -> None:
    model = StubChatModel(text_response="  That covers it, thanks.  ")
    interviewer = LLMInterviewer(model)
    transcript = [Turn(question="What happened?", answer="We migrated.")]

    closing = interviewer.closing_turn(transcript)

    assert closing == "That covers it, thanks."
    human_texts = [text for role, text in model.seen_inputs[0] if role == "human"]
    assert "Q: What happened?\nA: We migrated." in human_texts[0]

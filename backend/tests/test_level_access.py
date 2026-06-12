from pathlib import Path

from stubs import StubChatModel
from test_engine import build_engine, builder_returning, write_cv

from sotellme.interviewer import LLMInterviewer, Turn
from sotellme.profile import CandidateProfile, Role
from sotellme.role import CompetencyWeight, RoleContext

LEVEL_MARKERS = ("junior", "mid-level", "senior", "staff", "target level")

PROFILE = CandidateProfile(
    roles=[Role(title="Engineer", organization="Acme")],
    projects=[],
    quantified_claims=["Cut latency by 38%"],
    technologies=["Python"],
)

LEVELED_CONTEXT = RoleContext(
    company="Acme",
    role_title="Backend Engineer",
    competencies=[CompetencyWeight(name="ownership", weight=5)],
    target_level="staff",
)


def test_the_interviewer_prompt_assembly_never_carries_the_target_level() -> None:
    model = StubChatModel(text_response="A question.")
    interviewer = LLMInterviewer(model)
    transcript = [Turn(question="Tell me about the migration.", answer="We migrated.")]

    interviewer.competency_question(PROFILE, [], "ownership")
    interviewer.competency_question(PROFILE, transcript, "ownership")
    interviewer.probe_question(PROFILE, transcript, ("result",))
    interviewer.motivation_question(
        LEVELED_CONTEXT, "Backend Engineer at Acme.", transcript, "company"
    )
    interviewer.closing_turn(transcript)

    seen = " ".join(text for messages in model.seen_inputs for _, text in messages).lower()
    assert seen
    leaked = [marker for marker in LEVEL_MARKERS if marker in seen]
    assert not leaked, f"the interviewer saw level material: {leaked}"


def test_the_submitted_level_lands_in_state_for_the_grader_and_coach(tmp_path: Path) -> None:
    builder = builder_returning(LEVELED_CONTEXT.model_copy(update={"target_level": None}))
    with build_engine(tmp_path / "data", role_builder=builder) as engine:
        session = engine.start(write_cv(tmp_path), posting_text="Backend Engineer at Acme.")
        engine.submit_level(session.thread_id, "staff")
        state = engine._graph.get_state({"configurable": {"thread_id": session.thread_id}})

    assert state.values["role_context"].target_level == "staff"

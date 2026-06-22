"""Run one full interview session against the *installed* sotellme package.

This is the release smoke test: CI installs the built wheel into a clean
environment and runs this script. It drives the real engine — graph wiring,
SQLite checkpointing, report rendering — end to end, with stub agents in place
of the LLMs so it needs no API key and makes no network call. If the wheel is
missing a module, the entry point is broken, or a runtime dependency wasn't
declared, this fails where `--help` alone would pass.
"""

import tempfile
from collections.abc import Sequence
from pathlib import Path

from sotellme.assessor import AnswerAssessment, StarFlags
from sotellme.coach import AnswerAdvice, CoachReport, Drill
from sotellme.director import DirectorDecision, DirectorSituation
from sotellme.engine import InterviewEngine
from sotellme.grader import AnswerScore, SessionGrade
from sotellme.guardrail import GuardrailVerdict
from sotellme.interviewer import Turn
from sotellme.profile import CandidateProfile, Role
from sotellme.report import render_report
from sotellme.role import CompetencyWeight, RoleContext, TargetLevel

PROFILE = CandidateProfile(
    roles=[Role(title="Engineer", organization="Acme")],
    projects=[],
    quantified_claims=["Led the Acme migration"],
    technologies=["Python"],
)

CONTEXT = RoleContext(
    company="Acme",
    role_title="Senior Backend Engineer",
    competencies=[CompetencyWeight(name="ownership", weight=5)],
    target_level="senior",
)

GRADE = SessionGrade(
    scores=[
        AnswerScore(
            question="Tell me about their background.",
            turn_index=1,
            star=StarFlags(
                situation=True, task=True, action=True, result=True, quantified_result=True
            ),
            specificity="high",
            ownership="clear",
            weak_or_missing=[],
            gap="Single-team scope, one refinement short of a five.",
            rationale="Complete STAR with a measured outcome.",
            score=4,
        )
    ]
)

REPORT = CoachReport(
    summary="Solid, complete stories.",
    answer_advice=[
        AnswerAdvice(
            question="Tell me about their background.",
            diagnosis="You named the work and how it landed.",
            fix="Keep ending on the number you measured.",
        )
    ],
    drills=[Drill(focus="Stating results", exercise="Retell a project ending on a metric.")],
    study_plan="Turn each project into a STAR story that ends on a number.",
)


def parse_profile(cv_text: str) -> CandidateProfile:
    return PROFILE


def assess(topic: str, transcript: Sequence[Turn]) -> AnswerAssessment:
    return AnswerAssessment(
        star=StarFlags(situation=True, task=True, action=True, result=True, quantified_result=True),
        sufficient_signal=True,
        claims_worth_chasing=[],
    )


def build_role(posting_text: str) -> RoleContext:
    return CONTEXT


def research(posting_text: str, context: RoleContext) -> str:
    return "Acme builds billing software."


def grade(transcript: Sequence[Turn], target_level: TargetLevel) -> SessionGrade:
    return GRADE


def coach(
    transcript: Sequence[Turn], session_grade: SessionGrade, target_level: TargetLevel
) -> CoachReport:
    return REPORT


class ScriptedDirector:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, situation: DirectorSituation) -> DirectorDecision:
        self.calls += 1
        if self.calls == 1:
            return DirectorDecision(
                action="new_topic", subject="their background", reason="the opener"
            )
        return DirectorDecision(action="wrap_up", reason="enough signal")


class StubInterviewer:
    def question_for(
        self,
        decision: DirectorDecision,
        profile: CandidateProfile,
        context: RoleContext,
        brief: str,
        transcript: Sequence[Turn],
    ) -> str:
        return f"Tell me about {decision.subject}."

    def closing_turn(self, transcript: Sequence[Turn]) -> str:
        return "Thanks, that covers it."

    def redirect_turn(self, question: str) -> str:
        return question


class AllowGuardrail:
    def classify(self, question: str, answer: str) -> GuardrailVerdict:
        return "allow"


def run_smoke_session(data_dir: Path, cv_path: Path) -> str:
    with InterviewEngine(
        data_dir=data_dir,
        profile_parser=parse_profile,
        assessor=assess,
        director=ScriptedDirector(),
        interviewer=StubInterviewer(),
        role_builder=build_role,
        researcher=research,
        grader=grade,
        coacher=coach,
        guardrail=AllowGuardrail(),
    ) as engine:
        session = engine.start(cv_path)
        assert session.needs_level, "a session without a posting must ask for the target level"
        session = engine.submit_level(session.thread_id, "senior")
        assert session.question is not None, "the opener question should be posed after setup"
        result = engine.submit_answer(
            session.thread_id, "Situation, task, action, result, quantified — enough signal."
        )

    assert result.finished, "the scripted wrap-up should finish the session"
    assert result.closing, "a finished session should carry a closing turn"
    assert result.grade is not None, "a finished session should carry a grade"
    assert result.coach is not None, "a finished session should carry a coaching report"
    markdown = render_report(result.coach, result.transcript)
    assert markdown.strip(), "the rendered report should not be empty"
    return markdown


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cv_path = root / "cv.md"
        cv_path.write_text("# Jane Doe\nEngineer at Acme")
        run_smoke_session(root / "data", cv_path)
    print("smoke session: a full interview ran from the installed package and produced a report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared mock report fixtures for the task 0038 dev scripts.

Dev-time only, throwaway. A realistic senior session — transcript, grade, and coaching — used by
both `report_layouts_prototype.py` (layout chooser) and `report_view_check.py` (real-render check)
so the two never drift.
"""

from __future__ import annotations

from sotellme.assessor import StarFlags
from sotellme.coach import AnswerAdvice, CoachReport, Drill
from sotellme.grader import AnswerScore, SessionGrade, SkippedTurn
from sotellme.interviewer import Turn
from sotellme.profile import CandidateProfile, Role

LEVEL = "senior"

Q_SYSTEM = "Tell me about a system you designed end to end and the hardest tradeoff you made."
Q_OWNERSHIP = "Just to confirm — this was a service you owned, not one you inherited?"
Q_INCIDENT = "Describe a time you had to debug a production incident under pressure."
Q_DISAGREEMENT = "How did you handle a disagreement with another engineer on a design decision?"
Q_PERF = "Tell me about a time you improved the performance of something."
Q_REGRET = "What's a technical decision you regret, and what did you learn?"


def mock_profile() -> CandidateProfile:
    """A minimal candidate profile to anchor a mock session."""
    return CandidateProfile(
        roles=[Role(title="Engineer", organization="Acme")],
        projects=[],
        quantified_claims=[],
        technologies=[],
    )


def mock_transcript() -> list[Turn]:
    """A realistic six-turn transcript spanning strong and weak answers."""
    return [
        Turn(
            question=Q_SYSTEM,
            answer=(
                "At Acme I designed the billing reconciliation service. The hard tradeoff was "
                "consistency versus latency: I chose an eventually-consistent ledger with a "
                "nightly reconciliation pass, which let checkout stay under 200ms but meant "
                "finance saw numbers settle a few hours late. I wrote an ADR, walked finance "
                "through it, and we agreed the lag was acceptable below a $5k threshold."
            ),
        ),
        Turn(
            question=Q_OWNERSHIP,
            answer="Yes, I built it from scratch and owned it for about two years.",
        ),
        Turn(
            question=Q_INCIDENT,
            answer=(
                "We had an outage once. It was pretty stressful but the team came together and "
                "we fixed it. I helped out where I could and we learned a lot from it afterwards."
            ),
        ),
        Turn(
            question=Q_DISAGREEMENT,
            answer=(
                "A staff engineer wanted to put the rate limiter in the gateway; I thought it "
                "belonged in the service so we could shape limits per tenant. I built a small "
                "spike showing per-tenant limits were impossible at the gateway with our setup, "
                "shared the numbers, and we moved it into the service. We shipped it in the next "
                "sprint and tenant complaints dropped."
            ),
        ),
        Turn(
            question=Q_PERF,
            answer=(
                "I made a slow endpoint faster. I looked at it and found some queries that were "
                "doing too much, so I cleaned those up and it got better. Users were happier."
            ),
        ),
        Turn(
            question=Q_REGRET,
            answer=(
                "I pushed us onto a NoSQL store for a relational workload because it was the "
                "exciting choice. Six months in we were hand-rolling joins in application code. "
                "I led the migration back to Postgres, and now I default to boring tech and make "
                "the team write down why we'd deviate."
            ),
        ),
    ]


def mock_grade() -> SessionGrade:
    """Five scored answers spanning the 1-5 range, plus one skipped confirmation."""
    return SessionGrade(
        scores=[
            AnswerScore(
                question=Q_SYSTEM,
                turn_index=1,
                rationale="Concrete system, named tradeoff, quantified threshold, clear ownership.",
                star=StarFlags(
                    situation=True, task=True, action=True, result=True, quantified_result=True
                ),
                specificity="high",
                ownership="clear",
                weak_or_missing=[],
                gap="",
                score=5,
            ),
            AnswerScore(
                question=Q_INCIDENT,
                turn_index=3,
                rationale="No situation detail, no concrete action, credits the team, no outcome.",
                star=StarFlags(
                    situation=False, task=False, action=False, result=False, quantified_result=False
                ),
                specificity="low",
                ownership="unclear",
                weak_or_missing=["situation", "task", "action", "result"],
                gap="It never says what you personally did or how the incident actually resolved.",
                score=2,
            ),
            AnswerScore(
                question=Q_DISAGREEMENT,
                turn_index=4,
                rationale="Clear conflict, spike as evidence, named outcome, strong ownership.",
                star=StarFlags(
                    situation=True, task=True, action=True, result=True, quantified_result=False
                ),
                specificity="high",
                ownership="clear",
                weak_or_missing=["quantified_result"],
                gap="The outcome is real but unquantified — 'complaints dropped' by how much?",
                score=4,
            ),
            AnswerScore(
                question=Q_PERF,
                turn_index=5,
                rationale="Generic throughout: no system, no numbers, no specific change named.",
                star=StarFlags(
                    situation=False, task=True, action=False, result=False, quantified_result=False
                ),
                specificity="low",
                ownership="mixed",
                weak_or_missing=["situation", "action", "result", "quantified_result"],
                gap="Nothing concrete: which endpoint, which queries, how much faster?",
                score=2,
            ),
            AnswerScore(
                question=Q_REGRET,
                turn_index=6,
                rationale="Honest, specific failure, concrete recovery action, durable lesson.",
                star=StarFlags(
                    situation=True, task=True, action=True, result=True, quantified_result=False
                ),
                specificity="high",
                ownership="clear",
                weak_or_missing=["quantified_result"],
                gap="A migration cost is implied but never sized — how long, how much pain?",
                score=4,
            ),
        ],
        skipped=[
            SkippedTurn(
                turn_index=2,
                question=Q_OWNERSHIP,
                reason="a confirmation question with no STAR substance to grade",
            ),
        ],
    )


def mock_coach() -> CoachReport:
    """Coaching that names the two weak answers and the recurring specificity gap."""
    return CoachReport(
        summary=(
            "Your strongest answers — the billing system and the rate-limiter disagreement — are "
            "genuinely senior: a real tradeoff, an artifact behind the decision, and a clear line "
            "between what you did and what the team did. The problem is consistency. The incident "
            "and performance answers collapse into vague summary the moment there's no story you "
            "rehearsed, and at this level a thin answer reads as a gap rather than modesty. Make "
            "every answer carry one concrete system, one specific action, and one number."
        ),
        answer_advice=[
            AnswerAdvice(
                question=Q_INCIDENT,
                diagnosis=(
                    "This is the weakest answer in the set. 'We had an outage, the team came "
                    "together' tells me nothing about what you saw, what you did, or how it was "
                    "resolved. It credits the team and disappears you."
                ),
                fix=(
                    "Rebuild it as a single incident: what alerted you, the first hypothesis you "
                    "formed, the command or dashboard that confirmed it, the fix you shipped, and "
                    "the recovery time. Lead with 'I' for your part."
                ),
            ),
            AnswerAdvice(
                question=Q_PERF,
                diagnosis=(
                    "Every clause here is generic — 'a slow endpoint', 'some queries', 'it got "
                    "better'. There's a real story behind this; none of it survived into the reply."
                ),
                fix=(
                    "Name the endpoint, the p99 before, the specific query problem (N+1? missing "
                    "index?), the change, and the p99 after. One sentence of numbers beats a "
                    "paragraph of 'better'."
                ),
            ),
            AnswerAdvice(
                question=Q_DISAGREEMENT,
                diagnosis="Strong answer — spike as evidence is right. Only the outcome floats.",
                fix=(
                    "Close it with a number: tenant complaints dropped from X to Y, or "
                    "limit-related pages fell by Z%."
                ),
            ),
        ],
        drills=[
            Drill(
                focus="Quantifying outcomes",
                exercise=(
                    "Take your three best stories and write the closing line of each as a single "
                    "metric: before → after. No metric you can't defend; pick a real one each time."
                ),
            ),
            Drill(
                focus="Rescuing thin answers",
                exercise=(
                    "For any question where you don't have a rehearsed story, practice the "
                    "situation-action-result skeleton out loud in 30 seconds: one line each, "
                    "concrete, first-person."
                ),
            ),
        ],
        study_plan=(
            "First, fix the floor: rebuild the incident and performance answers so they carry a "
            "concrete system, your specific action, and a number — that's where you're losing the "
            "most. Second, attach a quantified outcome to the disagreement and regret answers, "
            "which are already strong. Drill quantifying outcomes daily this week; drill the thin-"
            "answer skeleton before your next real interview."
        ),
    )

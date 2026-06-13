import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field
from voice import voice_tells

from sotellme.config import PROVIDER_KEY_VARS, build_chat_model, resolve_model_config
from sotellme.director import DirectorDecision
from sotellme.interviewer import LLMInterviewer, Turn, render_transcript
from sotellme.profile import CandidateProfile
from sotellme.prompts import CLOSING_EXAMPLES, STYLE_EXAMPLES
from sotellme.role import RoleContext

CASES_FILE = Path(__file__).parent.parent / "evals" / "interviewer_cases.json"

EVAL_PROVIDER = os.environ.get("SOTELLME_PROVIDER", "google_genai")
JUDGE_PROVIDER = os.environ.get("SOTELLME_EVAL_JUDGE_PROVIDER", "anthropic")
JUDGE_MODEL = os.environ.get("SOTELLME_EVAL_JUDGE_MODEL") or None


def _missing_key(provider: str) -> bool:
    return not os.environ.get(PROVIDER_KEY_VARS.get(provider, ""), "")


needs_provider_key = pytest.mark.skipif(
    _missing_key(EVAL_PROVIDER) or _missing_key(JUDGE_PROVIDER),
    reason=f"interviewer evals need real {EVAL_PROVIDER} and {JUDGE_PROVIDER} keys",
)


def load_document() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(CASES_FILE.read_text())
    return document


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = load_document()["cases"]
    return cases


def decision_from(directive: dict[str, str]) -> DirectorDecision:
    return DirectorDecision.model_validate(directive)


class JudgeVerdict(BaseModel):
    chases_subject: bool = Field(
        description=(
            "True only if the question genuinely asks for the information the directive "
            "names as its subject."
        )
    )
    leads_witness: bool = Field(
        description=(
            "True if the question suggests what the answer should be, offers options to "
            "pick from, or folds the asker's own assumptions about what happened into "
            "the question."
        )
    )


JUDGE_SYSTEM_PROMPT = (
    "You evaluate a single follow-up question from a behavioral interview.\n"
    "You are given the interview transcript so far, the directive the question was "
    "written from (what to chase, and why), and the question itself.\n"
    "Judge the question strictly and return your verdict."
)

JUDGE_HUMAN_TEMPLATE = (
    "Transcript so far:\n<transcript>\n{transcript_text}\n</transcript>\n"
    "The directive the question was written from: chase {subject} (because {reason}).\n"
    "The question to judge:\n<question>\n{question}\n</question>"
)


def judge_question(question: str, transcript_text: str, directive: dict[str, str]) -> JudgeVerdict:
    config = resolve_model_config(env=os.environ, provider=JUDGE_PROVIDER, smart_model=JUDGE_MODEL)
    judge = build_chat_model(config, "smart").with_structured_output(JudgeVerdict)
    verdict = judge.invoke(
        [
            ("system", JUDGE_SYSTEM_PROMPT),
            (
                "human",
                JUDGE_HUMAN_TEMPLATE.format(
                    transcript_text=transcript_text,
                    subject=directive["subject"],
                    reason=directive["reason"],
                    question=question,
                ),
            ),
        ]
    )
    assert isinstance(verdict, JudgeVerdict)
    return verdict


def build_interviewer() -> LLMInterviewer:
    config = resolve_model_config(env=os.environ, provider=EVAL_PROVIDER)
    return LLMInterviewer(build_chat_model(config, "fast"))


def neutral_context() -> RoleContext:
    return RoleContext.model_validate(
        {"company": None, "role_title": None, "competencies": [{"name": "ownership", "weight": 3}]}
    )


def style_example_fingerprints() -> list[str]:
    fragments = [
        fragment.strip(" .,?").lower()
        for example in STYLE_EXAMPLES + CLOSING_EXAMPLES
        for fragment in re.split(r"\[[^\]]*\]", example)
    ]
    return [fragment for fragment in fragments if len(fragment.split()) >= 6]


def assert_clean_voice(text: str) -> None:
    assert not voice_tells(text), f"voice tells in {text!r}: {voice_tells(text)}"


def test_fingerprints_cover_the_style_examples() -> None:
    fingerprints = style_example_fingerprints()

    assert len(fingerprints) >= 2
    assert all("[" not in fingerprint for fingerprint in fingerprints)


def assert_no_style_example_leakage(question: str) -> None:
    assert "[" not in question and "]" not in question, (
        f"placeholder brackets leaked into the question: {question!r}"
    )
    lowered = question.lower()
    for fingerprint in style_example_fingerprints():
        assert fingerprint not in lowered, (
            f"style-example wording leaked into the question: {fingerprint!r} in {question!r}"
        )


@needs_provider_key
def test_an_opening_topic_directive_references_a_real_cv_claim() -> None:
    document = load_document()
    profile = CandidateProfile.model_validate(document["profile"])
    decision = decision_from(document["opening"]["directive"])

    question = build_interviewer().question_for(decision, profile, neutral_context(), "", [])

    substrings: list[str] = document["opening"]["claim_substrings"]
    assert any(claim in question.lower() for claim in substrings), (
        f"opening question references no profile claim: {question!r}"
    )
    assert_clean_voice(question)
    assert_no_style_example_leakage(question)


@needs_provider_key
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: str(case["name"]))
def test_the_follow_up_chases_the_directives_subject_without_leading(case: dict[str, Any]) -> None:
    document = load_document()
    profile = CandidateProfile.model_validate(document["profile"])
    transcript = [Turn.model_validate(turn) for turn in case["transcript"]]
    decision = decision_from(case["directive"])

    question = build_interviewer().question_for(
        decision, profile, neutral_context(), "", transcript
    )

    assert_clean_voice(question)
    assert_no_style_example_leakage(question)
    verdict = judge_question(question, render_transcript(transcript), case["directive"])
    assert verdict.chases_subject, (
        f"question does not chase {case['directive']['subject']!r}: {question!r}"
    )
    assert not verdict.leads_witness, f"question leads the witness: {question!r}"


@needs_provider_key
def test_a_minimal_profile_draws_no_content_from_the_style_examples() -> None:
    document = load_document()
    leakage = document["leakage"]
    profile = CandidateProfile.model_validate(leakage["profile"])
    transcript = [Turn.model_validate(turn) for turn in leakage["transcript"]]
    interviewer = build_interviewer()

    opening = interviewer.question_for(
        DirectorDecision(action="new_topic", subject="their background", reason="the opener"),
        profile,
        neutral_context(),
        "",
        [],
    )
    probe = interviewer.question_for(
        decision_from(leakage["directive"]), profile, neutral_context(), "", transcript
    )

    for question in (opening, probe):
        assert_clean_voice(question)
        assert_no_style_example_leakage(question)


@needs_provider_key
def test_a_new_topic_directive_opens_plainly_without_reading_the_cv_back() -> None:
    document = load_document()
    opener = document["topic_opener"]
    profile = CandidateProfile.model_validate(document["profile"])
    transcript = [Turn.model_validate(turn) for turn in opener["transcript"]]

    question = build_interviewer().question_for(
        decision_from(opener["directive"]), profile, neutral_context(), "", transcript
    )

    lowered = question.lower()
    assert "what was going on that made" not in lowered, (
        f"question reaches for the contorted opener: {question!r}"
    )
    verbatim_reads: list[str] = opener["cv_verbatim_substrings"]
    leaked = [substring for substring in verbatim_reads if substring in lowered]
    assert not leaked, f"question reads the CV line back ({leaked}): {question!r}"
    assert "ferryline" in lowered or "ferry" in lowered, (
        f"opener never anchors in the project it opens: {question!r}"
    )
    assert_clean_voice(question)
    assert_no_style_example_leakage(question)


@needs_provider_key
def test_a_company_directive_names_the_actual_product_or_domain() -> None:
    document = load_document()
    personalization = document["personalization"]
    profile = CandidateProfile.model_validate(document["profile"])
    context = RoleContext.model_validate(personalization["role_context"])
    transcript = [Turn.model_validate(turn) for turn in personalization["transcript"]]

    question = build_interviewer().question_for(
        decision_from(personalization["directive"]),
        profile,
        context,
        personalization["brief"],
        transcript,
    )

    substrings: list[str] = personalization["product_substrings"]
    assert any(substring in question.lower() for substring in substrings), (
        f"company question never names the product or domain: {question!r}"
    )
    assert_clean_voice(question)
    assert_no_style_example_leakage(question)


@needs_provider_key
def test_the_closing_turn_signs_off_without_asking_or_judging() -> None:
    document = load_document()
    transcript = [Turn.model_validate(turn) for turn in document["closing"]["transcript"]]

    closing = build_interviewer().closing_turn(transcript)

    assert closing
    assert "?" not in closing, f"the closing turn asks a question: {closing!r}"
    assert_clean_voice(closing)
    assert_no_style_example_leakage(closing)

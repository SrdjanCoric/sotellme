import json
import os
import re
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field
from voice import voice_tells

from sotellme.config import PROVIDER_KEY_VARS, build_chat_model, resolve_model_config
from sotellme.coverage import Gap
from sotellme.interviewer import LLMInterviewer, Turn, render_transcript
from sotellme.profile import CandidateProfile
from sotellme.prompts import GAP_GUIDANCE, STYLE_EXAMPLES

CASES_FILE = Path(__file__).parent.parent / "evals" / "interviewer_cases.json"

EVAL_PROVIDER = os.environ.get("SOTELLME_PROVIDER", "anthropic")

needs_provider_key = pytest.mark.skipif(
    not os.environ.get(PROVIDER_KEY_VARS.get(EVAL_PROVIDER, ""), ""),
    reason=f"interviewer evals need a real {EVAL_PROVIDER} key",
)


def load_document() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(CASES_FILE.read_text())
    return document


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = load_document()["cases"]
    return cases


class JudgeVerdict(BaseModel):
    probes_gap: bool = Field(
        description=(
            "True only if the question genuinely asks for the missing information "
            "described by the gap."
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
    "You are given the interview transcript so far, a description of the information gap "
    "the question was meant to target, and the question itself.\n"
    "Judge the question strictly and return your verdict."
)

JUDGE_HUMAN_TEMPLATE = (
    "Transcript so far:\n<transcript>\n{transcript_text}\n</transcript>\n"
    "The gap the question was meant to target: {gap_guidance}\n"
    "The question to judge:\n<question>\n{question}\n</question>"
)


def judge_question(question: str, transcript_text: str, gap_guidance: str) -> JudgeVerdict:
    config = resolve_model_config(env=os.environ, provider=EVAL_PROVIDER)
    judge = build_chat_model(config, "smart").with_structured_output(JudgeVerdict)
    verdict = judge.invoke(
        [
            ("system", JUDGE_SYSTEM_PROMPT),
            (
                "human",
                JUDGE_HUMAN_TEMPLATE.format(
                    transcript_text=transcript_text,
                    gap_guidance=gap_guidance,
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


def style_example_fingerprints() -> list[str]:
    fragments = [
        fragment.strip(" .,?").lower()
        for example in STYLE_EXAMPLES
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
def test_the_opening_question_references_a_real_cv_claim() -> None:
    document = load_document()
    profile = CandidateProfile.model_validate(document["profile"])

    question = build_interviewer().opening_question(profile)

    substrings: list[str] = document["opening"]["claim_substrings"]
    assert any(claim in question.lower() for claim in substrings), (
        f"opening question references no profile claim: {question!r}"
    )
    assert_clean_voice(question)
    assert_no_style_example_leakage(question)


@needs_provider_key
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: str(case["name"]))
def test_the_probe_targets_the_flagged_gap_without_leading(case: dict[str, Any]) -> None:
    document = load_document()
    profile = CandidateProfile.model_validate(document["profile"])
    transcript = [Turn.model_validate(turn) for turn in case["transcript"]]
    gaps = cast(tuple[Gap, ...], tuple(case["gaps"]))

    question = build_interviewer().probe_question(profile, transcript, gaps)

    assert_clean_voice(question)
    assert_no_style_example_leakage(question)
    verdict = judge_question(question, render_transcript(transcript), GAP_GUIDANCE[gaps[0]])
    assert verdict.probes_gap, f"question does not probe the {gaps[0]} gap: {question!r}"
    assert not verdict.leads_witness, f"question leads the witness: {question!r}"


@needs_provider_key
def test_a_minimal_profile_draws_no_content_from_the_style_examples() -> None:
    document = load_document()
    leakage = document["leakage"]
    profile = CandidateProfile.model_validate(leakage["profile"])
    transcript = [Turn.model_validate(turn) for turn in leakage["transcript"]]
    gaps = cast(tuple[Gap, ...], tuple(leakage["gaps"]))
    interviewer = build_interviewer()

    opening = interviewer.opening_question(profile)
    probe = interviewer.probe_question(profile, transcript, gaps)

    for question in (opening, probe):
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

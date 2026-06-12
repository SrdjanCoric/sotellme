import json
import os
from pathlib import Path
from typing import Any

import pytest

from sotellme.config import PROVIDER_KEY_VARS, build_chat_model, resolve_model_config
from sotellme.extraction import extract_cv_text
from sotellme.profile import CandidateProfile, parse_candidate_profile

CASES_FILE = Path(__file__).parent.parent / "evals" / "profile_parser_cases.json"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

EVAL_PROVIDER = os.environ.get("SOTELLME_PROVIDER", "anthropic")

needs_provider_key = pytest.mark.skipif(
    not os.environ.get(PROVIDER_KEY_VARS.get(EVAL_PROVIDER, ""), ""),
    reason=f"profile-parser evals need a real {EVAL_PROVIDER} key",
)


def load_cases() -> list[dict[str, Any]]:
    document = json.loads(CASES_FILE.read_text())
    cases: list[dict[str, Any]] = document["cases"]
    return cases


def field_texts(profile: CandidateProfile, field: str) -> list[str]:
    if field == "roles":
        return [f"{r.title} {r.organization} {r.period or ''}" for r in profile.roles]
    if field == "projects":
        return [f"{p.name} {p.description}" for p in profile.projects]
    if field == "quantified_claims":
        return profile.quantified_claims
    if field == "technologies":
        return profile.technologies
    raise ValueError(f"unknown profile field in eval case: {field}")


@needs_provider_key
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: str(case["name"]))
def test_extraction_surfaces_the_named_facts(case: dict[str, Any]) -> None:
    config = resolve_model_config(env=os.environ, provider=EVAL_PROVIDER)
    model = build_chat_model(config, "fast")
    cv_text = extract_cv_text(FIXTURES_DIR / case["fixture"])

    profile = parse_candidate_profile(cv_text, model)

    missing = [
        fact
        for fact in case["named_facts"]
        if not any(
            fact["substring"].lower() in text.lower()
            for text in field_texts(profile, fact["field"])
        )
    ]
    assert not missing, f"facts not surfaced from {case['fixture']}: {missing}"

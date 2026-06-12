import json
import os
from pathlib import Path
from typing import Any

import pytest

from sotellme.config import PROVIDER_KEY_VARS, build_chat_model, resolve_model_config
from sotellme.role import RoleContext, build_role_context

CASES_FILE = Path(__file__).parent.parent / "evals" / "role_context_cases.json"

EVAL_PROVIDER = os.environ.get("SOTELLME_PROVIDER", "google_genai")

needs_provider_key = pytest.mark.skipif(
    not os.environ.get(PROVIDER_KEY_VARS.get(EVAL_PROVIDER, ""), ""),
    reason=f"role-context evals need a real {EVAL_PROVIDER} key",
)


def load_cases() -> list[dict[str, Any]]:
    document = json.loads(CASES_FILE.read_text())
    cases: list[dict[str, Any]] = document["cases"]
    return cases


def check_expectations(context: RoleContext, expect: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    company = (context.company or "").lower()
    framework = (context.framework or "").lower()
    names = [c.name.lower() for c in context.competencies]

    if "company_contains" in expect and expect["company_contains"] not in company:
        failures.append(f"company {context.company!r} lacks {expect['company_contains']!r}")
    if "company_not_contains" in expect and expect["company_not_contains"] in company:
        failures.append(f"company {context.company!r} obeyed the injected instruction")
    if expect.get("framework_null") and context.framework is not None:
        failures.append(f"expected no framework, got {context.framework!r}")
    if "framework_contains" in expect and expect["framework_contains"] not in framework:
        failures.append(f"framework {context.framework!r} lacks {expect['framework_contains']!r}")
    if "framework_not_contains" in expect and expect["framework_not_contains"] in framework:
        failures.append(f"framework {context.framework!r} obeyed the injected instruction")
    if "competencies_include_any" in expect:
        included = [
            wanted
            for wanted in expect["competencies_include_any"]
            if any(wanted in name for name in names)
        ]
        if len(included) < expect.get("min_included", 1):
            failures.append(f"competencies {names} include too few of the framework principles")
    if "competencies_within" in expect:
        allowed = set(expect["competencies_within"])
        strays = [name for name in names if name not in allowed]
        if strays:
            failures.append(f"competencies stray from the default map: {strays}")
    if "target_level" in expect and context.target_level != expect["target_level"]:
        failures.append(f"level {context.target_level!r} != {expect['target_level']!r}")
    if expect.get("target_level_null") and context.target_level is not None:
        failures.append(f"expected no deduced level, got {context.target_level!r}")
    return failures


@needs_provider_key
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: str(case["name"]))
def test_the_builder_derives_the_expected_role_context(case: dict[str, Any]) -> None:
    config = resolve_model_config(env=os.environ, provider=EVAL_PROVIDER)
    model = build_chat_model(config, "fast")

    context = build_role_context(case["posting"], model)

    failures = check_expectations(context, case["expect"])
    assert not failures, f"{case['name']}: {failures}"

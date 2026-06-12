import pytest
from pydantic import ValidationError
from stubs import StubChatModel

from sotellme.profile import (
    CandidateProfile,
    ProfileParseError,
    Project,
    Role,
    parse_candidate_profile,
)

SAMPLE_PROFILE = CandidateProfile(
    roles=[Role(title="Senior Engineer", organization="Acme")],
    projects=[],
    quantified_claims=["Cut latency by 38%"],
    technologies=["Python"],
)


def test_profile_with_a_role_validates() -> None:
    profile = CandidateProfile(
        roles=[Role(title="Senior Engineer", organization="Acme")],
        projects=[],
        quantified_claims=["Cut latency by 38%"],
        technologies=["Python"],
    )

    assert profile.roles[0].title == "Senior Engineer"


def test_profile_with_only_projects_validates() -> None:
    profile = CandidateProfile(
        roles=[],
        projects=[Project(name="openroster", description="Shift-planning library")],
        quantified_claims=[],
        technologies=[],
    )

    assert profile.projects[0].name == "openroster"


def test_profile_with_no_roles_and_no_projects_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one role or project"):
        CandidateProfile(roles=[], projects=[], quantified_claims=[], technologies=[])


def test_parser_returns_the_extracted_profile() -> None:
    model = StubChatModel(structured_response=SAMPLE_PROFILE)

    profile = parse_candidate_profile("# Jane Doe\nSenior Engineer at Acme", model)

    assert profile == SAMPLE_PROFILE


def test_parser_sends_the_cv_to_the_model_as_delimited_data() -> None:
    model = StubChatModel(structured_response=SAMPLE_PROFILE)

    parse_candidate_profile("# Jane Doe\nSenior Engineer at Acme", model)

    human_texts = [text for role, text in model.seen_inputs[0] if role == "human"]
    assert "<cv>\n# Jane Doe\nSenior Engineer at Acme\n</cv>" in human_texts[0]


def _validation_error() -> ValidationError:
    try:
        CandidateProfile(roles=[], projects=[], quantified_claims=[], technologies=[])
    except ValidationError as exc:
        return exc
    raise AssertionError("expected the empty profile to fail validation")


def test_extraction_that_fails_validation_is_a_clear_error() -> None:
    model = StubChatModel(structured_error=_validation_error())

    with pytest.raises(ProfileParseError, match="Could not extract a structured profile"):
        parse_candidate_profile("not really a cv", model)


def test_extraction_that_returns_nothing_is_a_clear_error() -> None:
    model = StubChatModel(structured_response=None)

    with pytest.raises(ProfileParseError, match="Could not extract a structured profile"):
        parse_candidate_profile("not really a cv", model)

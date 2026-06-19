"""Extract a structured candidate profile from CV text using a chat model."""

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError, model_validator

from sotellme.caching import cache_system_prompt
from sotellme.prompts import profile_extraction_messages


class ProfileParseError(Exception):
    """Raised when a structured candidate profile cannot be extracted from the CV."""


class Role(BaseModel):
    """A professional role held by the candidate as stated in the CV."""

    title: str = Field(description="Job title as stated in the CV.")
    organization: str = Field(description="Employer or organization name.")
    period: str | None = Field(default=None, description="Time period of the role, if stated.")


class Project(BaseModel):
    """A notable project, professional or personal, drawn from the CV."""

    name: str = Field(description="Project name as stated in the CV.")
    description: str = Field(description="What the project is and the candidate's part in it.")


class CandidateProfile(BaseModel):
    """Structured profile extracted from a candidate's CV."""

    roles: list[Role] = Field(description="Professional roles held by the candidate.")
    projects: list[Project] = Field(description="Notable projects, professional or personal.")
    quantified_claims: list[str] = Field(
        description=(
            "Verbatim claims from the CV that carry a number or measurable outcome, "
            "e.g. 'reduced p95 latency by 38%' or 'mentored a team of 6 engineers'. "
            "Include team sizes, counts, percentages, durations, and money amounts."
        )
    )
    technologies: list[str] = Field(description="Technologies, languages, and tools the CV names.")

    @model_validator(mode="after")
    def _require_substance(self) -> "CandidateProfile":
        """Validate that the profile has at least one role or project."""
        if not self.roles and not self.projects:
            raise ValueError("a profile needs at least one role or project")
        return self


_PARSE_FAILURE_MESSAGE = (
    "Could not extract a structured profile from the CV. "
    "Check that the file really is a CV and try again."
)


def parse_candidate_profile(
    cv_text: str, model: BaseChatModel, provider: str = ""
) -> CandidateProfile:
    """Extract a structured candidate profile from CV text.

    Args:
        cv_text: The raw text of the candidate's CV.
        model: The chat model used for structured extraction.
        provider: Provider name used to tailor system-prompt caching.

    Returns:
        The extracted candidate profile.

    Raises:
        ProfileParseError: If extraction fails validation or does not return a profile.
    """
    structured = model.with_structured_output(CandidateProfile)
    try:
        result = structured.invoke(
            cache_system_prompt(profile_extraction_messages(cv_text), provider)
        )
    except (ValidationError, OutputParserException) as exc:
        raise ProfileParseError(_PARSE_FAILURE_MESSAGE) from exc
    if not isinstance(result, CandidateProfile):
        raise ProfileParseError(_PARSE_FAILURE_MESSAGE)
    return result

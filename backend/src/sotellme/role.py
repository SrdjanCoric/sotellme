"""Derive a weighted role context from a job posting using a chat model."""

from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError, model_validator

from sotellme.caching import cache_system_prompt
from sotellme.prompts import role_context_messages

TargetLevel = Literal["junior", "mid", "senior", "staff"]

DEFAULT_COMPETENCIES = ("ownership", "impact", "conflict", "failure", "ambiguity")

DEFAULT_WEIGHT = 3


_LEVEL_EMPHASIS_LADDER: tuple[tuple[TargetLevel, tuple[str, ...]], ...] = (
    ("junior", ("problem solving", "delivery", "learning")),
    ("mid", ("initiative", "trust and conflict")),
    ("senior", ("strategic leadership", "developing others", "innovation")),
    ("staff", ()),
)


def level_emphasis(level: TargetLevel) -> tuple[str, ...]:
    """Return the cumulative emphasis themes for a target seniority level.

    Accumulates the emphasis themes of each rung of the ladder up to and including
    the given level.

    Args:
        level: The target seniority level.

    Returns:
        The emphasis themes accumulated from the lowest rung up to the given level.
        Returns all themes if the level is not found on the ladder.
    """
    emphasis: tuple[str, ...] = ()
    for rung, names in _LEVEL_EMPHASIS_LADDER:
        emphasis += names
        if rung == level:
            return emphasis
    return emphasis


class CompetencyWeight(BaseModel):
    """A competency the role values, weighted by the posting's emphasis."""

    name: str = Field(description="The competency or principle the role values.")
    weight: int = Field(
        ge=1,
        le=5,
        description="How heavily the posting emphasizes this competency, 1 (barely) to 5 (core).",
    )


class RoleContext(BaseModel):
    """Structured context about a role derived from its job posting."""

    company: str | None = Field(default=None, description="Company name, if the posting names it.")
    role_title: str | None = Field(
        default=None, description="The role title as the posting states it."
    )
    competencies: list[CompetencyWeight] = Field(
        description="Competencies the role values, weighted by the posting's emphasis."
    )
    framework: str | None = Field(
        default=None,
        description=(
            "The published values framework the round maps onto, if the company has one, "
            "e.g. 'Amazon Leadership Principles'. Null when there is none."
        ),
    )
    target_level: TargetLevel | None = Field(
        default=None,
        description=(
            "Seniority the posting targets, only when it states one explicitly through "
            "a level word in the title or a years-of-experience requirement. "
            "Null whenever there is no explicit signal."
        ),
    )

    @model_validator(mode="after")
    def _require_competencies(self) -> "RoleContext":
        """Validate that the role context has at least one competency."""
        if not self.competencies:
            raise ValueError("a role context needs at least one competency")
        return self


class RoleContextError(Exception):
    """Raised when a role context cannot be derived from a job posting."""


_BUILD_FAILURE_MESSAGE = (
    "Could not derive a role context from the job posting. "
    "Check that the text really is a posting and try again."
)


def build_role_context(posting_text: str, model: BaseChatModel, provider: str = "") -> RoleContext:
    """Derive a structured role context from a job posting.

    Args:
        posting_text: The raw text of the job posting.
        model: The chat model used for structured extraction.
        provider: Provider name used to tailor system-prompt caching.

    Returns:
        The derived role context.

    Raises:
        RoleContextError: If extraction fails validation or does not return a context.
    """
    structured = model.with_structured_output(RoleContext)
    try:
        result = structured.invoke(
            cache_system_prompt(role_context_messages(posting_text), provider)
        )
    except (ValidationError, OutputParserException) as exc:
        raise RoleContextError(_BUILD_FAILURE_MESSAGE) from exc
    if not isinstance(result, RoleContext):
        raise RoleContextError(_BUILD_FAILURE_MESSAGE)
    return result


def default_role_context() -> RoleContext:
    """Build a generic role context using the default competencies and weight.

    Returns:
        A role context populated with the default competencies at the default weight.
    """
    return RoleContext(
        competencies=[
            CompetencyWeight(name=name, weight=DEFAULT_WEIGHT) for name in DEFAULT_COMPETENCIES
        ],
    )

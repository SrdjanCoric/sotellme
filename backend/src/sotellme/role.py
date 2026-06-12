from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError, model_validator

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
    emphasis: tuple[str, ...] = ()
    for rung, names in _LEVEL_EMPHASIS_LADDER:
        emphasis += names
        if rung == level:
            return emphasis
    return emphasis


class CompetencyWeight(BaseModel):
    name: str = Field(description="The competency or principle the role values.")
    weight: int = Field(
        ge=1,
        le=5,
        description="How heavily the posting emphasizes this competency, 1 (barely) to 5 (core).",
    )


class RoleContext(BaseModel):
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
        if not self.competencies:
            raise ValueError("a role context needs at least one competency")
        return self


class RoleContextError(Exception):
    pass


_BUILD_FAILURE_MESSAGE = (
    "Could not derive a role context from the job posting. "
    "Check that the text really is a posting and try again."
)


def build_role_context(posting_text: str, model: BaseChatModel) -> RoleContext:
    structured = model.with_structured_output(RoleContext)
    try:
        result = structured.invoke(role_context_messages(posting_text))
    except (ValidationError, OutputParserException) as exc:
        raise RoleContextError(_BUILD_FAILURE_MESSAGE) from exc
    if not isinstance(result, RoleContext):
        raise RoleContextError(_BUILD_FAILURE_MESSAGE)
    return result


def default_role_context() -> RoleContext:
    return RoleContext(
        competencies=[
            CompetencyWeight(name=name, weight=DEFAULT_WEIGHT) for name in DEFAULT_COMPETENCIES
        ],
    )

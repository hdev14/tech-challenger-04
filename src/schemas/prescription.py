from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.common import Evidence, SourcedText


class Medication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name_as_written: SourcedText
    normalized_name: SourcedText | None = None
    strength: SourcedText | None = None
    form: SourcedText | None = None
    dose: SourcedText | None = None
    route: SourcedText | None = None
    frequency: SourcedText | None = None
    duration: SourcedText | None = None
    dates: SourcedText | None = None
    instructions: SourcedText | None = None
    indication: SourcedText | None = None
    uncertainties: list[str] = Field(default_factory=list)


class ModelPrescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary_items: list[SourcedText] = Field(min_length=1)
    medications: list[Medication]
    uncertainties: list[str] = Field(default_factory=list)


class PrescriptionAnalysis(BaseModel):
    status: Literal["completed", "needs_review"]
    summary: str
    summary_evidence: list[Evidence] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    pages_with_text: list[int] = Field(default_factory=list)
    pages_without_text: list[int] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    requires_review: Literal[True] = True

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

JobKind = Literal["audio", "prescription"]
JobStatus = Literal["queued", "processing", "completed", "needs_review", "failed"]
Locale = Literal["pt-BR", "en-US"]


class JobError(BaseModel):
    code: str
    message: str


class JobOutcome(BaseModel):
    status: Literal["completed", "needs_review"]
    analysis_id: str
    error: JobError | None = None


class UploadAccepted(BaseModel):
    job_id: str
    case_id: str
    status: Literal["queued"] = "queued"
    status_url: str


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=1)
    quote: Annotated[str, Field(min_length=1, max_length=500)]

    @field_validator("quote")
    @classmethod
    def quote_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A citação não pode estar vazia.")
        return value


class SourcedText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Annotated[str, Field(min_length=1, max_length=500)]
    evidence: Evidence

    @field_validator("value")
    @classmethod
    def value_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("O valor não pode estar vazio.")
        return value


class AlertDraft(BaseModel):
    type: str
    source: Literal["audio", "prescription"]
    message: str
    evidence: dict[str, Any]
    requires_review: Literal[True] = True

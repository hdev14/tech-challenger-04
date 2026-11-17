from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.common import Locale

MentionType = Literal[
    "symptom_mentioned",
    "condition_mentioned",
    "medication_mentioned",
    "clinical_mention",
]


class AudioSegment(BaseModel):
    text: str
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)


class AudioEvidence(BaseModel):
    source: Literal["audio"] = "audio"
    segment_index: int = Field(ge=0)
    quote: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    text_offset: int | None = Field(default=None, ge=0)
    text_length: int | None = Field(default=None, ge=0)


class Assertion(BaseModel):
    certainty: str | None = None
    conditionality: str | None = None
    association: str | None = None


class ClinicalMention(BaseModel):
    type: MentionType
    text: str
    category: str
    normalized_text: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    assertion: Assertion | None = None
    evidence: AudioEvidence
    requires_review: Literal[True] = True


class RelationRole(BaseModel):
    name: str
    text: str
    category: str
    text_offset: int | None = None


class ClinicalRelation(BaseModel):
    type: str
    roles: list[RelationRole]
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: AudioEvidence
    requires_review: Literal[True] = True


class CriticalTerm(BaseModel):
    text: str
    assertion: Assertion | None = None
    confidence: float | None = None
    evidence: AudioEvidence
    requires_review: Literal[True] = True


class SentimentSegment(BaseModel):
    sentiment: str
    confidence: dict[str, float]
    evidence: AudioEvidence
    requires_review: Literal[True] = True


class LanguageStep(BaseModel):
    status: Literal["available", "partial", "unavailable", "disabled"]
    reason: str | None = None


class SentimentStep(LanguageStep):
    segments: list[SentimentSegment] = Field(default_factory=list)


class AudioAnalysis(BaseModel):
    status: Literal["completed", "needs_review"]
    language: Locale
    transcript: str
    segments: list[AudioSegment]
    clinical_analysis: LanguageStep
    clinical_entities: list[ClinicalMention] = Field(default_factory=list)
    relations: list[ClinicalRelation] = Field(default_factory=list)
    relevant_terms: list[CriticalTerm] = Field(default_factory=list)
    text_sentiment: SentimentStep
    acoustic_features: LanguageStep = Field(
        default_factory=lambda: LanguageStep(status="disabled")
    )
    uncertainties: list[str] = Field(default_factory=list)
    requires_review: Literal[True] = True

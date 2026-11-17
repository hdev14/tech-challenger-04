from typing import Any, Literal

from pydantic import BaseModel

from src.schemas.audio import AudioAnalysis
from src.schemas.common import JobError, JobKind, JobStatus
from src.schemas.prescription import PrescriptionAnalysis


class AnalysisReference(BaseModel):
    analysis_id: str
    analysis_url: str
    analysis: AudioAnalysis | PrescriptionAnalysis


class JobResponse(BaseModel):
    job_id: str
    case_id: str
    kind: JobKind
    status: JobStatus
    result: AnalysisReference | None = None
    error: JobError | None = None


class AnalysisResponse(BaseModel):
    analysis_id: str
    case_id: str
    document_id: str
    modality: JobKind
    result: AudioAnalysis | PrescriptionAnalysis
    created_at: str


class AlertResponse(BaseModel):
    alert_id: str
    analysis_id: str
    type: str
    source: JobKind
    message: str
    evidence: dict[str, Any]
    review_status: Literal["pending"]
    requires_review: Literal[True]
    created_at: str


class AlertListResponse(BaseModel):
    case_id: str
    limit: int
    offset: int
    items: list[AlertResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"]

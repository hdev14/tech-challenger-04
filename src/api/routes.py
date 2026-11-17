from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Path as ApiPath
from fastapi import Query, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from src.core.config import Settings
from src.database import SQLiteRepository
from src.jobs.runtime import RedisJobQueue
from src.schemas.api import AlertListResponse, AnalysisResponse, HealthResponse, JobResponse
from src.schemas.common import JobKind, Locale, UploadAccepted
from src.services.upload import InvalidUpload, StoredUpload, UploadTooLarge, store_upload
from src.services.upload import validate_audio, validate_pdf

router = APIRouter(prefix="/api/v1")
CaseId = Annotated[
    str, ApiPath(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
]


async def _submit(
    case_id: str,
    kind: JobKind,
    file: UploadFile,
    language: Locale,
    request: Request,
) -> UploadAccepted:
    settings: Settings = request.app.state.settings
    repository: SQLiteRepository = request.app.state.repository
    jobs: RedisJobQueue = request.app.state.jobs
    suffix = ".wav" if kind == "audio" else ".pdf"
    if not file.filename or Path(file.filename).suffix.lower() != suffix:
        await file.close()
        raise HTTPException(status_code=400, detail=f"Envie um arquivo {suffix}.")

    max_bytes = settings.max_audio_bytes if kind == "audio" else settings.max_pdf_bytes
    try:
        upload: StoredUpload = await store_upload(file, settings.upload_dir, max_bytes, suffix)
    except UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidUpload as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    scheduled = False
    try:
        metadata: dict[str, Any]
        if kind == "audio":
            audio_metadata = await run_in_threadpool(
                validate_audio, upload.path, settings.max_audio_duration_seconds
            )
            metadata = {**audio_metadata, "language": language}
        else:
            metadata = await run_in_threadpool(validate_pdf, upload.path, settings.max_pdf_pages)
        document_id = await run_in_threadpool(
            repository.create_document,
            case_id,
            kind,
            upload.sha256,
            upload.size_bytes,
            metadata,
        )
        job_id = await run_in_threadpool(
            jobs.enqueue,
            case_id,
            document_id,
            kind,
            upload.path,
            language,
        )
        scheduled = True
        return UploadAccepted(
            job_id=job_id,
            case_id=case_id,
            status_url=f"/api/v1/jobs/{job_id}",
        )
    except InvalidUpload as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if not scheduled:
            upload.path.unlink(missing_ok=True)


@router.post(
    "/cases/{case_id}/audio",
    response_model=UploadAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enviar áudio WAV para transcrição e análise",
)
async def upload_audio(
    case_id: CaseId,
    request: Request,
    file: Annotated[UploadFile, File()],
    language: Annotated[Locale, Form()] = "en-US",
) -> UploadAccepted:
    return await _submit(case_id, "audio", file, language, request)


@router.post(
    "/cases/{case_id}/prescriptions",
    response_model=UploadAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enviar PDF de prescrição para extração textual",
)
async def upload_prescription(
    case_id: CaseId,
    request: Request,
    file: Annotated[UploadFile, File()],
) -> UploadAccepted:
    return await _submit(case_id, "prescription", file, "en-US", request)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    jobs: RedisJobQueue = request.app.state.jobs
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    return job


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: str, request: Request) -> dict[str, Any]:
    repository: SQLiteRepository = request.app.state.repository
    analysis = repository.get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Análise não encontrada.")
    return analysis


@router.get("/cases/{case_id}/alerts", response_model=AlertListResponse)
def get_alerts(
    case_id: CaseId,
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    repository: SQLiteRepository = request.app.state.repository
    return {
        "case_id": case_id,
        "limit": limit,
        "offset": offset,
        "items": repository.list_alerts(case_id, limit, offset),
    }


@router.get("/health", response_model=HealthResponse)
def health() -> dict[str, str]:
    return {"status": "ok"}

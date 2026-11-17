import logging
import traceback
from pathlib import Path
from typing import Any

from rq import get_current_job

from src.core.config import Settings
from src.core.errors import ProcessingError
from src.database import SQLiteRepository
from src.schemas.audio import AudioAnalysis
from src.schemas.common import JobError, JobKind, JobOutcome, Locale
from src.schemas.prescription import PrescriptionAnalysis
from src.services.alert_service import audio_alerts, prescription_alerts
from src.services.audio_pipeline import process_audio
from src.services.prescription_pipeline import process_prescription

logger = logging.getLogger(__name__)


class JobProcessingFailed(RuntimeError):
    pass


def _log_failure(job_id: str, exc: Exception) -> None:
    frames = traceback.extract_tb(exc.__traceback__)
    locations = " > ".join(f"{frame.name}:{frame.lineno}" for frame in frames)
    logger.error("Job %s falhou (%s); pilha: %s", job_id, type(exc).__name__, locations)


def process_job(
    case_id: str,
    document_id: str,
    kind: JobKind,
    path: str,
    language: Locale,
    database_path: str,
) -> dict[str, Any]:
    job = get_current_job()
    if job is None:
        raise RuntimeError("O processamento precisa ser executado por um worker RQ.")

    upload_path = Path(path)
    outcome: JobOutcome | None = None
    error: JobError | None = None
    try:
        settings = Settings()
        repository = SQLiteRepository(Path(database_path))
        analysis: AudioAnalysis | PrescriptionAnalysis
        if kind == "audio":
            audio = process_audio(upload_path, language, settings)
            analysis = audio
            alerts = audio_alerts(audio)
        else:
            prescription = process_prescription(upload_path, settings)
            analysis = prescription
            alerts = prescription_alerts(prescription)
        analysis_id = repository.save_analysis(case_id, document_id, kind, analysis, alerts)
        outcome = JobOutcome(status=analysis.status, analysis_id=analysis_id)
    except ProcessingError as exc:
        _log_failure(job.id, exc)
        error = JobError(code=exc.code, message=exc.message)
    except Exception as exc:
        _log_failure(job.id, exc)
        error = JobError(
            code="PROCESSING_FAILED",
            message="Não foi possível processar o documento. Consulte os logs seguros.",
        )
    finally:
        try:
            upload_path.unlink(missing_ok=True)
        except OSError as exc:
            _log_failure(job.id, exc)
            error = JobError(
                code="TEMP_FILE_CLEANUP_FAILED",
                message="Não foi possível remover o arquivo temporário; verifique o armazenamento local.",
            )
            if outcome is not None:
                outcome.status = "needs_review"

    if outcome is not None:
        outcome.error = error
        return outcome.model_dump(mode="json")

    if error is None:
        raise RuntimeError("O processamento terminou sem resultado ou erro.")
    job.meta["error"] = error.model_dump(mode="json")
    job.save_meta()
    raise JobProcessingFailed(error.message) from None

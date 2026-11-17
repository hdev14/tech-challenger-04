import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from redis import Redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus as RQJobStatus
from rq.serializers import JSONSerializer

from src.core.config import Settings
from src.database import SQLiteRepository
from src.jobs.processor import process_job
from src.schemas.common import JobError, JobKind, JobOutcome, Locale


def create_redis_connection(settings: Settings) -> Redis:
    return Redis.from_url(
        settings.redis_url.get_secret_value(),
        decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


class RedisJobQueue:
    def __init__(self, settings: Settings, repository: SQLiteRepository) -> None:
        self.settings = settings
        self.repository = repository
        self.connection = create_redis_connection(settings)
        self.queue = Queue(
            settings.rq_queue_name,
            connection=self.connection,
            serializer=JSONSerializer,
            default_timeout=settings.rq_job_timeout_seconds,
        )

    def enqueue(
        self,
        case_id: str,
        document_id: str,
        kind: JobKind,
        path: Path,
        language: Locale,
    ) -> str:
        job = self.queue.enqueue_call(
            process_job,
            args=(
                case_id,
                document_id,
                kind,
                str(path.resolve()),
                language,
                str(self.repository.path.resolve()),
            ),
            job_id=uuid4().hex,
            result_ttl=self.settings.rq_result_ttl_seconds,
            failure_ttl=self.settings.rq_failure_ttl_seconds,
            meta={"case_id": case_id, "kind": kind},
            description="process_document",
        )
        return job.id

    def get(self, job_id: str) -> dict[str, Any] | None:
        if re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
            return None
        try:
            job = Job.fetch(job_id, connection=self.connection, serializer=JSONSerializer)
        except NoSuchJobError:
            return None
        if job.origin != self.queue.name or job.func_name != "src.jobs.processor.process_job":
            return None

        case_id = job.meta["case_id"]
        kind = job.meta["kind"]
        rq_status = job.get_status(refresh=False)
        result: dict[str, Any] | None = None
        error: JobError | None = None
        if rq_status in (
            RQJobStatus.CREATED,
            RQJobStatus.QUEUED,
            RQJobStatus.DEFERRED,
            RQJobStatus.SCHEDULED,
        ):
            status = "queued"
        elif rq_status == RQJobStatus.STARTED:
            status = "processing"
        elif rq_status == RQJobStatus.FINISHED:
            stored_outcome = job.return_value()
            if stored_outcome is None and not self.connection.exists(job.key):
                return None
            outcome = JobOutcome.model_validate(stored_outcome)
            analysis = self.repository.get_analysis(outcome.analysis_id)
            if analysis is None or analysis["case_id"] != case_id or analysis["modality"] != kind:
                raise RuntimeError("A análise vinculada ao job não está disponível.")
            status = outcome.status
            result = {
                "analysis_id": outcome.analysis_id,
                "analysis_url": f"/api/v1/analyses/{outcome.analysis_id}",
                "analysis": analysis["result"],
            }
            error = outcome.error
        elif rq_status in (RQJobStatus.FAILED, RQJobStatus.STOPPED, RQJobStatus.CANCELED):
            status = "failed"
            error = (
                JobError.model_validate(job.meta["error"])
                if "error" in job.meta
                else JobError(
                    code="WORKER_FAILED",
                    message="O processamento foi interrompido. Consulte os logs seguros.",
                )
            )
        else:
            raise RuntimeError("O job possui um estado RQ desconhecido.")

        return {
            "job_id": job.id,
            "case_id": case_id,
            "kind": kind,
            "status": status,
            "result": result,
            "error": error,
        }

    def close(self) -> None:
        self.connection.close()

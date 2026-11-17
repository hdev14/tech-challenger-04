import sqlite3
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pymupdf
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from rq import SimpleWorker, Worker
from rq.job import JobStatus as RQJobStatus
from rq.serializers import JSONSerializer

from src.core.config import Settings
from src.database import SQLiteRepository
from src.jobs.processor import JobProcessingFailed, process_job
from src.jobs.runtime import RedisJobQueue
from src.jobs.worker import SpawnWorkerPool, main as start_worker
from src.main import create_app
from src.schemas.common import AlertDraft
from src.schemas.prescription import PrescriptionAnalysis


def test_sqlite_persists_analysis_and_alerts_without_jobs_table(tmp_path):
    path = tmp_path / "data.sqlite3"
    repository = SQLiteRepository(path)
    repository.initialize()
    document_id = repository.create_document("case-01", "prescription", "checksum", 42, {})
    result = PrescriptionAnalysis(
        status="needs_review",
        summary="",
        pages_without_text=[1],
        uncertainties=["Página sem texto"],
    )
    alert = AlertDraft(
        type="prescription_page_without_text",
        source="prescription",
        message="Revisar página",
        evidence={"page": 1},
    )
    analysis_id = repository.save_analysis("case-01", document_id, "prescription", result, [alert])

    reopened = SQLiteRepository(path)
    assert reopened.get_analysis(analysis_id)["result"]["uncertainties"] == ["Página sem texto"]
    assert reopened.list_alerts("case-01", 10, 0)[0]["evidence"]["page"] == 1
    assert reopened.list_alerts("outro", 10, 0) == []
    assert reopened.get_analysis("desconhecida") is None
    with sqlite3.connect(path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"cases", "documents", "analyses", "alerts"} <= tables
    assert "jobs" not in tables


def test_repository_rejects_nonpersistent_memory_database():
    with pytest.raises(ValueError):
        SQLiteRepository(Path(":memory:"))


@pytest.mark.parametrize("decode_responses", ["true", "false"])
def test_redis_url_cannot_override_binary_responses(decode_responses):
    with pytest.raises(ValidationError, match="decode_responses"):
        Settings(
            _env_file=None,
            redis_url=f"redis://127.0.0.1:6379/0?decode_responses={decode_responses}",
        )


def test_worker_fails_fast_if_redis_is_unavailable(monkeypatch):
    connection = MagicMock()
    connection.ping.side_effect = RedisConnectionError("indisponível")
    pool = MagicMock()
    monkeypatch.setattr("src.jobs.worker.Settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr("src.jobs.worker.create_redis_connection", lambda _settings: connection)
    monkeypatch.setattr("src.jobs.worker.WorkerPool", pool)

    with pytest.raises(RedisConnectionError):
        start_worker()
    pool.assert_not_called()
    connection.close.assert_called_once()


@pytest.mark.parametrize(
    ("platform_name", "use_spawn", "worker_class"),
    [("darwin", True, SimpleWorker), ("linux", False, Worker)],
)
def test_worker_selects_process_model_for_platform(monkeypatch, platform_name, use_spawn, worker_class):
    settings = Settings(_env_file=None)
    connection = MagicMock()
    spawn_pool = MagicMock()
    default_pool = MagicMock()
    monkeypatch.setattr("src.jobs.worker.Settings", lambda: settings)
    monkeypatch.setattr("src.jobs.worker.create_redis_connection", lambda _settings: connection)
    monkeypatch.setattr("src.jobs.worker.sys", SimpleNamespace(platform=platform_name))
    monkeypatch.setattr("src.jobs.worker.SpawnWorkerPool", spawn_pool)
    monkeypatch.setattr("src.jobs.worker.WorkerPool", default_pool)

    start_worker()

    selected, other = (spawn_pool, default_pool) if use_spawn else (default_pool, spawn_pool)
    selected.assert_called_once()
    other.assert_not_called()
    assert selected.call_args.args == ([settings.rq_queue_name],)
    assert selected.call_args.kwargs["num_workers"] == settings.max_concurrent_jobs
    assert selected.call_args.kwargs["worker_class"] is worker_class
    selected.return_value.start.assert_called_once()
    connection.close.assert_called_once()


def test_spawn_worker_pool_creates_spawn_process():
    connection = Redis.from_url("redis://127.0.0.1:6379/0", decode_responses=False)
    try:
        pool = SpawnWorkerPool(
            ["test"],
            connection=connection,
            worker_class=SimpleWorker,
            serializer=JSONSerializer,
        )
        process = pool.get_worker_process("worker-test", burst=True)
        assert isinstance(process, get_context("spawn").Process)
    finally:
        connection.close()


def test_job_expiring_during_result_lookup_returns_not_found(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, redis_url="redis://127.0.0.1:1/0")
    jobs = RedisJobQueue(settings, SQLiteRepository(tmp_path / "data.sqlite3"))
    job = MagicMock()
    job.origin = settings.rq_queue_name
    job.func_name = "src.jobs.processor.process_job"
    job.meta = {"case_id": "case-01", "kind": "prescription"}
    job.get_status.return_value = RQJobStatus.FINISHED
    job.return_value.return_value = None
    monkeypatch.setattr("src.jobs.runtime.Job.fetch", MagicMock(return_value=job))
    monkeypatch.setattr(jobs.connection, "exists", MagicMock(return_value=0))
    try:
        assert jobs.get("a" * 32) is None
    finally:
        jobs.close()


def test_analysis_and_alerts_are_saved_atomically(tmp_path):
    repository = SQLiteRepository(tmp_path / "data.sqlite3")
    repository.initialize()
    document_id = repository.create_document("case-01", "prescription", "checksum", 42, {})
    result = PrescriptionAnalysis(status="completed", summary="")
    invalid_alert = AlertDraft(
        type="invalid",
        source="prescription",
        message="Revisar",
        evidence={"not_json_serializable": object()},
    )

    with pytest.raises(TypeError):
        repository.save_analysis("case-01", document_id, "prescription", result, [invalid_alert])
    with sqlite3.connect(repository.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0] == 0


def test_processing_failure_is_safe_and_removes_temporary_file(tmp_path, monkeypatch, caplog):
    repository = SQLiteRepository(tmp_path / "data.sqlite3")
    repository.initialize()
    document_id = repository.create_document("case-01", "audio", "checksum", 1, {})
    job = MagicMock()
    job.id = "job-id"
    job.meta = {}
    path = tmp_path / "upload.wav"
    path.write_bytes(b"synthetic")

    def fail(*_args):
        raise RuntimeError("private transcript and provider response")

    monkeypatch.setattr("src.jobs.processor.get_current_job", lambda: job)
    monkeypatch.setattr("src.jobs.processor.Settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr("src.jobs.processor.process_audio", fail)
    with pytest.raises(JobProcessingFailed):
        process_job("case-01", document_id, "audio", str(path), "pt-BR", str(repository.path))

    assert job.meta["error"]["code"] == "PROCESSING_FAILED"
    job.save_meta.assert_called_once()
    assert "private transcript" not in caplog.text
    assert not path.exists()


def test_job_persists_result_and_review_alert_before_cleanup(tmp_path, monkeypatch):
    repository = MagicMock(spec=SQLiteRepository)
    repository.save_analysis.return_value = "analysis-id"
    job = MagicMock()
    job.id = "job-id"
    path = tmp_path / "upload.pdf"
    path.write_bytes(b"synthetic")
    analysis = PrescriptionAnalysis(
        status="needs_review",
        summary="",
        pages_without_text=[1],
        uncertainties=["Página sem texto"],
    )
    monkeypatch.setattr("src.jobs.processor.get_current_job", lambda: job)
    monkeypatch.setattr("src.jobs.processor.Settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr("src.jobs.processor.SQLiteRepository", lambda *_args: repository)
    monkeypatch.setattr("src.jobs.processor.process_prescription", lambda *_args: analysis)

    outcome = process_job(
        "case-01", "document-id", "prescription", str(path), "en-US", "data.sqlite3"
    )
    saved = repository.save_analysis.call_args.args
    assert saved[:4] == ("case-01", "document-id", "prescription", analysis)
    assert saved[4][0].evidence["page"] == 1
    assert outcome == {"status": "needs_review", "analysis_id": "analysis-id", "error": None}
    assert not path.exists()


def test_cleanup_failure_keeps_analysis_and_requires_review(tmp_path, monkeypatch):
    repository = MagicMock(spec=SQLiteRepository)
    repository.save_analysis.return_value = "analysis-id"
    job = MagicMock()
    job.id = "job-id"
    path = tmp_path / "upload.pdf"
    path.write_bytes(b"synthetic")
    analysis = PrescriptionAnalysis(status="completed", summary="")
    original_unlink = Path.unlink

    def fail_upload_cleanup(candidate, *args, **kwargs):
        if candidate == path:
            raise OSError("private filesystem detail")
        return original_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr("src.jobs.processor.get_current_job", lambda: job)
    monkeypatch.setattr("src.jobs.processor.Settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr("src.jobs.processor.SQLiteRepository", lambda *_args: repository)
    monkeypatch.setattr("src.jobs.processor.process_prescription", lambda *_args: analysis)
    monkeypatch.setattr(Path, "unlink", fail_upload_cleanup)

    outcome = process_job(
        "case-01", "document-id", "prescription", str(path), "en-US", "data.sqlite3"
    )
    assert outcome["status"] == "needs_review"
    assert outcome["analysis_id"] == "analysis-id"
    assert outcome["error"]["code"] == "TEMP_FILE_CLEANUP_FAILED"
    assert path.exists()


def test_redis_failure_returns_503_and_cleans_upload(tmp_path, monkeypatch, caplog):
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "data.sqlite3",
        upload_dir=tmp_path / "uploads",
        redis_url="redis://127.0.0.1:1/0",
    )
    app = create_app(settings)
    with pymupdf.open() as document:
        document.new_page()
        pdf = document.tobytes()

    with TestClient(app) as client:
        failure = RedisConnectionError("private connection details")
        monkeypatch.setattr(app.state.jobs.queue, "enqueue_call", MagicMock(side_effect=failure))
        response = client.post(
            "/api/v1/cases/case-01/prescriptions",
            files={"file": ("sample.pdf", pdf, "application/pdf")},
        )
        assert response.status_code == 503
        assert response.json() == {"detail": "Fila de jobs indisponível."}
        assert list(settings.upload_dir.iterdir()) == []

        monkeypatch.setattr("src.jobs.runtime.Job.fetch", MagicMock(side_effect=failure))
        response = client.get("/api/v1/jobs/" + "a" * 32)
        assert response.status_code == 503
        assert "private connection details" not in response.text
        assert "private connection details" not in caplog.text

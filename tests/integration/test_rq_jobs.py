import io
import os
import subprocess
import sys
import time
import wave
from contextlib import contextmanager
from multiprocessing import current_process, get_context
from pathlib import Path
from uuid import uuid4

import pymupdf
import pytest
from fastapi.testclient import TestClient
from redis import Redis
from rq import Queue, Worker
from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus
from rq.serializers import JSONSerializer

from src.core.config import Settings
from src.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def worker_execution_context():
    return {
        "pid": os.getpid(),
        "spawned": isinstance(current_process(), get_context("spawn").Process),
    }


@pytest.fixture
def redis_setup(tmp_path):
    url = os.environ.get("TEST_REDIS_URL")
    if not url:
        pytest.skip("Configure TEST_REDIS_URL para testar com um Redis real.")
    connection = Redis.from_url(url, decode_responses=False)
    connection.ping()
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "data.sqlite3",
        upload_dir=tmp_path / "uploads",
        redis_url=url,
        rq_queue_name=f"test-{uuid4().hex}",
        max_concurrent_jobs=1,
    )
    job_ids = []
    yield settings, connection, job_ids
    for job_id in job_ids:
        try:
            Job.fetch(job_id, connection=connection, serializer=JSONSerializer).delete()
        except NoSuchJobError:
            pass
        connection.delete(f"rq:results:{job_id}")
    Queue(settings.rq_queue_name, connection=connection).delete(delete_jobs=True)
    connection.close()


def pdf_bytes(text=""):
    with pymupdf.open() as document:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
        return document.tobytes()


def audio_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8_000)
        audio.writeframes(b"\0\0" * 800)
    return buffer.getvalue()


@contextmanager
def running_worker(settings, connection, tmp_path):
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "PYTHONPATH": os.pathsep.join((str(PROJECT_ROOT), str(Path(__file__).resolve().parent))),
        "REDIS_URL": settings.redis_url.get_secret_value(),
        "RQ_QUEUE_NAME": settings.rq_queue_name,
        "MAX_CONCURRENT_JOBS": str(settings.max_concurrent_jobs),
    }
    log_path = tmp_path / "worker.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "src.jobs.worker"],
            cwd=tmp_path,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            queue = Queue(settings.rq_queue_name, connection=connection)
            deadline = time.monotonic() + 20
            while Worker.count(queue=queue) < settings.max_concurrent_jobs:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise AssertionError(f"Worker não iniciou: {log_path.read_text()}")
                time.sleep(0.1)
            yield process, log_path
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise AssertionError("O pool de workers não encerrou corretamente.")


def wait_for_terminal_job(client, job_id, process, log_path):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in ("completed", "needs_review", "failed"):
            return job
        if process.poll() is not None:
            break
        time.sleep(0.1)
    raise AssertionError(f"Job não finalizou: {log_path.read_text()}")


def test_real_redis_jobs_survive_api_restart_and_fail_safely(redis_setup, tmp_path):
    settings, connection, job_ids = redis_setup
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/cases/case-01/prescriptions",
            files={"file": ("blank.pdf", pdf_bytes(), "application/pdf")},
        )
        assert response.status_code == 202
        accepted = response.json()
        job_ids.append(accepted["job_id"])
        assert accepted["status"] == "queued"
        assert client.get(accepted["status_url"]).json()["status"] == "queued"

    queued = Job.fetch(job_ids[0], connection=connection, serializer=JSONSerializer)
    assert queued.get_status() == JobStatus.QUEUED
    assert queued.timeout == settings.rq_job_timeout_seconds
    assert queued.result_ttl == settings.rq_result_ttl_seconds
    assert queued.failure_ttl == settings.rq_failure_ttl_seconds

    with TestClient(create_app(settings)) as client:
        assert client.get(accepted["status_url"]).json()["status"] == "queued"
        with running_worker(settings, connection, tmp_path) as (process, log_path):
            completed = wait_for_terminal_job(client, job_ids[0], process, log_path)
            assert completed["status"] == "needs_review"
            assert completed["result"]["analysis"]["pages_without_text"] == [1]
            assert len(client.get("/api/v1/cases/case-01/alerts").json()["items"]) == 1
            assert list(settings.upload_dir.iterdir()) == []

            response = client.post(
                "/api/v1/cases/case-01/prescriptions",
                files={"file": ("text.pdf", pdf_bytes("private synthetic text"), "application/pdf")},
            )
            assert response.status_code == 202
            job_ids.append(response.json()["job_id"])
            failed = wait_for_terminal_job(client, job_ids[1], process, log_path)
            assert failed["status"] == "failed"
            assert failed["error"]["code"] == "AZURE_OPENAI_NOT_CONFIGURED"
            assert "private synthetic text" not in str(failed)
            assert list(settings.upload_dir.iterdir()) == []

            response = client.post(
                "/api/v1/cases/case-01/audio",
                files={"file": ("sample.wav", audio_bytes(), "audio/wav")},
                data={"language": "pt-BR"},
            )
            assert response.status_code == 202
            job_ids.append(response.json()["job_id"])
            failed_audio = wait_for_terminal_job(client, job_ids[2], process, log_path)
            assert failed_audio["status"] == "failed"
            assert failed_audio["error"]["code"] == "AZURE_SPEECH_NOT_CONFIGURED"
            assert list(settings.upload_dir.iterdir()) == []

        assert "private synthetic text" not in log_path.read_text()

    finished_job = Job.fetch(job_ids[0], connection=connection, serializer=JSONSerializer)
    assert finished_job.get_status() == JobStatus.FINISHED
    assert finished_job.return_value()["analysis_id"] == completed["result"]["analysis_id"]
    assert "analysis" not in finished_job.return_value()
    assert 0 < connection.ttl(finished_job.key) <= settings.rq_result_ttl_seconds
    failed_job = Job.fetch(job_ids[1], connection=connection, serializer=JSONSerializer)
    assert failed_job.get_status() == JobStatus.FAILED
    assert 0 < connection.ttl(failed_job.key) <= settings.rq_failure_ttl_seconds

    with TestClient(create_app(settings)) as client:
        assert client.get(accepted["status_url"]).json() == completed
        assert client.get("/api/v1/jobs/" + "f" * 32).status_code == 404


def test_worker_pool_runs_configured_number_of_jobs_concurrently(redis_setup, tmp_path):
    settings, connection, job_ids = redis_setup
    settings.max_concurrent_jobs = 2
    queue = Queue(settings.rq_queue_name, connection=connection, serializer=JSONSerializer)
    for _ in range(2):
        job = queue.enqueue_call("time.sleep", args=(2,), result_ttl=60)
        job_ids.append(job.id)

    with running_worker(settings, connection, tmp_path) as (process, log_path):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            statuses = [
                Job.fetch(job_id, connection=connection, serializer=JSONSerializer).get_status()
                for job_id in job_ids
            ]
            if statuses == [JobStatus.STARTED, JobStatus.STARTED]:
                break
            if process.poll() is not None:
                break
            time.sleep(0.05)
        else:
            raise AssertionError(f"Jobs não executaram simultaneamente: {log_path.read_text()}")
        assert statuses == [JobStatus.STARTED, JobStatus.STARTED]


@pytest.mark.skipif(sys.platform != "darwin", reason="O processo sem fork é específico do macOS.")
def test_macos_worker_spawns_without_fork_and_handles_timeout(redis_setup, tmp_path):
    settings, connection, job_ids = redis_setup
    queue = Queue(settings.rq_queue_name, connection=connection, serializer=JSONSerializer)
    with running_worker(settings, connection, tmp_path) as (process, log_path):
        workers = Worker.all(queue=queue, connection=connection)
        assert len(workers) == 1
        worker_pid = workers[0].pid

        def run_job(func, args=(), timeout=None):
            job = queue.enqueue_call(func, args=args, timeout=timeout, result_ttl=60, failure_ttl=60)
            job_ids.append(job.id)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                job.refresh()
                if job.get_status() in (JobStatus.FINISHED, JobStatus.FAILED):
                    return job
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            raise AssertionError(f"Job não finalizou: {log_path.read_text()}")

        context = run_job("test_rq_jobs.worker_execution_context")
        assert context.get_status() == JobStatus.FINISHED
        assert context.return_value() == {"pid": worker_pid, "spawned": True}

        timed_out = run_job("time.sleep", args=(3,), timeout=1)
        assert timed_out.get_status() == JobStatus.FAILED
        timeout_result = timed_out.latest_result()
        assert timeout_result is not None
        assert timeout_result.exc_string is not None
        assert "JobTimeoutException" in timeout_result.exc_string

        next_job = run_job("test_rq_jobs.worker_execution_context")
        assert next_job.get_status() == JobStatus.FINISHED
        assert next_job.return_value() == {"pid": worker_pid, "spawned": True}


def test_job_expiry_does_not_remove_persisted_analysis(redis_setup, tmp_path):
    settings, connection, job_ids = redis_setup
    settings.rq_result_ttl_seconds = 2
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/cases/expiry/prescriptions",
            files={"file": ("blank.pdf", pdf_bytes(), "application/pdf")},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        job_ids.append(job_id)
        with running_worker(settings, connection, tmp_path) as (process, log_path):
            completed = wait_for_terminal_job(client, job_id, process, log_path)
            assert completed["status"] == "needs_review"
            analysis_id = completed["result"]["analysis_id"]
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if client.get(f"/api/v1/jobs/{job_id}").status_code == 404:
                    break
                time.sleep(0.1)
            else:
                raise AssertionError("O job não expirou no Redis.")
            assert client.get(f"/api/v1/analyses/{analysis_id}").status_code == 200
            assert client.get("/api/v1/cases/expiry/alerts").json()["items"]

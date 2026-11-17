import sys
from multiprocessing import get_context
from multiprocessing.process import BaseProcess

from rq import SimpleWorker, Worker
from rq.serializers import JSONSerializer
from rq.worker_pool import WorkerPool, run_worker

from src.core.config import Settings
from src.jobs.runtime import create_redis_connection


class SpawnWorkerPool(WorkerPool):
    def get_worker_process(
        self,
        name: str,
        burst: bool,
        _sleep: float = 0,
        logging_level: str = "INFO",
    ) -> BaseProcess:
        # O WorkerPool do RQ escolhe fork explicitamente, mesmo no macOS.
        return get_context("spawn").Process(
            target=run_worker,
            args=(
                name,
                self._queue_names,
                self._connection_class,
                self._pool_class,
                self._pool_kwargs,
            ),
            kwargs={
                "_sleep": _sleep,
                "burst": burst,
                "logging_level": logging_level,
                "worker_class": self.worker_class,
                "job_class": self.job_class,
                "serializer": self.serializer,
                "exception_handlers": self.exception_handlers,
            },
            name=f"Worker {name} (WorkerPool {self.name})",
        )


def main() -> None:
    settings = Settings()
    connection = create_redis_connection(settings)
    try:
        connection.ping()
        is_macos = sys.platform == "darwin"
        pool_class = SpawnWorkerPool if is_macos else WorkerPool
        pool = pool_class(
            [settings.rq_queue_name],
            connection=connection,
            num_workers=settings.max_concurrent_jobs,
            serializer=JSONSerializer,
            worker_class=SimpleWorker if is_macos else Worker,
        )
        pool.start()
    finally:
        connection.close()


if __name__ == "__main__":
    main()

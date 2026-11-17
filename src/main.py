import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from src.api.routes import router
from src.core.config import Settings
from src.database import SQLiteRepository
from src.jobs.runtime import RedisJobQueue

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    configuration = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        repository = SQLiteRepository(configuration.database_path)
        repository.initialize()
        configuration.upload_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        app.state.settings = configuration
        app.state.repository = repository
        jobs = RedisJobQueue(configuration, repository)
        app.state.jobs = jobs
        try:
            yield
        finally:
            jobs.close()

    app = FastAPI(
        title="API Multimodal de Apoio à Revisão Clínica",
        version="1.0.0",
        description=(
            "Protótipo acadêmico sem autenticação. Resultados são menções e evidências, "
            "não diagnósticos ou recomendações; revisão profissional é necessária."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(RedisError)
    async def redis_unavailable(_request: Request, exc: RedisError) -> JSONResponse:
        logger.error("Fila Redis indisponível (%s).", type(exc).__name__)
        return JSONResponse(status_code=503, content={"detail": "Fila de jobs indisponível."})

    app.include_router(router)
    return app


app = create_app()

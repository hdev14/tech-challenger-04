from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    azure_speech_key: SecretStr | None = None
    azure_speech_region: str | None = Field(default=None, pattern=r"^[a-z0-9-]+$")
    azure_language_endpoint: str | None = None
    azure_language_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_deployment: str | None = None

    database_path: Path = Path(".runtime/multimodal.sqlite3")
    upload_dir: Path = Path(".runtime/uploads")
    redis_url: SecretStr = SecretStr("redis://127.0.0.1:6379/0")
    rq_queue_name: str = Field(
        default="default", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
    )
    rq_job_timeout_seconds: int = Field(default=1_800, gt=0)
    rq_result_ttl_seconds: int = Field(default=3_600, gt=0)
    rq_failure_ttl_seconds: int = Field(default=86_400, gt=0)
    max_audio_bytes: int = Field(default=20_000_000, gt=0)
    max_audio_duration_seconds: float = Field(default=300, gt=0)
    max_pdf_bytes: int = Field(default=10_000_000, gt=0)
    max_pdf_pages: int = Field(default=50, gt=0)
    max_pdf_text_chars: int = Field(default=200_000, gt=0)
    pdf_model_chunk_chars: int = Field(default=16_000, ge=1_000)
    max_concurrent_jobs: int = Field(default=2, ge=1, le=16)
    azure_request_timeout_seconds: float = Field(default=120, ge=1, le=600)
    speech_timeout_seconds: float = Field(default=420, ge=1, le=3_600)
    sentiment_enabled: bool = False
    critical_terms: list[str] = Field(
        default_factory=lambda: [
            "dor no peito",
            "falta de ar",
            "chest pain",
            "shortness of breath",
        ]
    )

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, url: SecretStr) -> SecretStr:
        query = parse_qs(urlsplit(url.get_secret_value()).query, keep_blank_values=True)
        if "decode_responses" in query:
            raise ValueError("Não configure decode_responses na URL do Redis.")
        return url

    @field_validator("critical_terms")
    @classmethod
    def validate_critical_terms(cls, terms: list[str]) -> list[str]:
        cleaned = [term.strip() for term in terms]
        if len(cleaned) > 100 or any(not term or len(term) > 100 for term in cleaned):
            raise ValueError("Os termos críticos devem ter entre 1 e 100 caracteres.")
        return list(dict.fromkeys(cleaned))

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from src.schemas.common import AlertDraft, JobKind


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteRepository:
    def __init__(self, path: Path) -> None:
        if str(path) == ":memory:":
            raise ValueError("O banco SQLite precisa de um caminho persistente em disco.")
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    modality TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    received_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    document_id TEXT NOT NULL REFERENCES documents(document_id),
                    modality TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    analysis_id TEXT NOT NULL REFERENCES analyses(analysis_id),
                    type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS alerts_case_created
                    ON alerts(case_id, created_at, alert_id);
                """
            )

    def create_document(
        self,
        case_id: str,
        modality: JobKind,
        sha256: str,
        size_bytes: int,
        metadata: dict[str, Any],
    ) -> str:
        document_id = uuid4().hex
        now = utc_now()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO cases (case_id, created_at) VALUES (?, ?)",
                (case_id, now),
            )
            connection.execute(
                """INSERT INTO documents
                   (document_id, case_id, modality, sha256, size_bytes, metadata_json, received_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    document_id,
                    case_id,
                    modality,
                    sha256,
                    size_bytes,
                    json.dumps(metadata, ensure_ascii=False, allow_nan=False),
                    now,
                ),
            )
        return document_id

    def save_analysis(
        self,
        case_id: str,
        document_id: str,
        modality: JobKind,
        result: BaseModel,
        alerts: list[AlertDraft],
    ) -> str:
        analysis_id = uuid4().hex
        now = utc_now()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """INSERT INTO analyses
                   (analysis_id, case_id, document_id, modality, result_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    analysis_id,
                    case_id,
                    document_id,
                    modality,
                    result.model_dump_json(),
                    now,
                ),
            )
            connection.executemany(
                """INSERT INTO alerts
                   (alert_id, case_id, analysis_id, type, source, message,
                    evidence_json, review_status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        uuid4().hex,
                        case_id,
                        analysis_id,
                        alert.type,
                        alert.source,
                        alert.message,
                        json.dumps(alert.evidence, ensure_ascii=False, allow_nan=False),
                        "pending",
                        now,
                    )
                    for alert in alerts
                ],
            )
        return analysis_id

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT analysis_id, case_id, document_id, modality, result_json, created_at
                   FROM analyses WHERE analysis_id = ?""",
                (analysis_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "analysis_id": row["analysis_id"],
            "case_id": row["case_id"],
            "document_id": row["document_id"],
            "modality": row["modality"],
            "result": json.loads(row["result_json"]),
            "created_at": row["created_at"],
        }

    def list_alerts(self, case_id: str, limit: int, offset: int) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT alert_id, analysis_id, type, source, message, evidence_json,
                          review_status, created_at
                   FROM alerts WHERE case_id = ?
                   ORDER BY created_at, alert_id LIMIT ? OFFSET ?""",
                (case_id, limit, offset),
            ).fetchall()
        return [
            {
                "alert_id": row["alert_id"],
                "analysis_id": row["analysis_id"],
                "type": row["type"],
                "source": row["source"],
                "message": row["message"],
                "evidence": json.loads(row["evidence_json"]),
                "review_status": row["review_status"],
                "requires_review": True,
                "created_at": row["created_at"],
            }
            for row in rows
        ]

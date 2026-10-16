"""P14 / GitHub PR15: submit and inspect durable queued transcription jobs.

No inference, worker, browser callback or separate FastAPI process is started.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Sequence
from uuid import UUID, uuid4

from backend.db.task.migrations import migrate_task_database
from modules.jobs.inputs import (
    InputLimits, InputSource, InputValidationError, validate_input,
)
from modules.jobs.storage import DEFAULT_JOB_ROOT, JobStorage, JobStorageError, StoredAudio
from modules.mvp.preset import MVP_PRESET


class JobIntakeError(RuntimeError):
    """Stable code without leaking paths, SQL or personal recording details."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class JobTicket:
    identifier: str
    status: str
    stage: str | None
    file_name: str
    model_name: str
    language: str
    audio_duration: float
    created_at: str


class JobService:
    """Single-process intake contract for future worker/UI adapters.

    Only accept InputSource, not paths directly. Allowed roots are trusted
    server config, never browser-submitted directories.
    """

    def __init__(
        self, *, storage_root: str | Path = DEFAULT_JOB_ROOT,
        upload_roots: Sequence[str | Path] = (),
        local_roots: Sequence[str | Path] = (),
        limits: InputLimits | None = None,
        ffprobe_binary: str = "ffprobe",
    ):
        self.storage = JobStorage(storage_root)
        self.upload_roots = tuple(upload_roots)
        self.local_roots = tuple(local_roots)
        self.limits = limits if limits is not None else InputLimits()
        if not isinstance(self.limits, InputLimits):
            raise TypeError("limits must be InputLimits")
        self.ffprobe_binary = ffprobe_binary
        if self.storage.database_path.is_symlink():
            raise JobIntakeError("UNSAFE_DATABASE", "Job database cannot be a symlink")
        migrate_task_database(self.storage.database_path)
        # Separate from the legacy tracked backend/records.db by construction.

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.storage.database_path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _insert_queued(self, job: StoredAudio, created_at: str) -> None:
        # The file has already been atomically published; QUEUED is committed last.
        params = json.dumps({
            "model_size": MVP_PRESET.model_size,
            "language": MVP_PRESET.language,
            "task": MVP_PRESET.task,
            "vad": MVP_PRESET.vad_enabled,
            "alignment": MVP_PRESET.alignment_enabled,
            "diarization": MVP_PRESET.diarization_enabled,
        }, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""
                INSERT INTO tasks
                (uuid, status, stage, task_type, result_type, file_name,
                 input_path, audio_duration, language, task_params, model_name,
                 attempt, progress, created_at, updated_at)
                VALUES (?, 'QUEUED', 'QUEUED', 'TRANSCRIPTION', 'JSON', ?,
                        ?, ?, ?, ?, ?, 0, 0.0, ?, ?)
            """, (
                job.job_id, job.source.safe_filename, job.relative_path,
                job.source.duration_seconds, MVP_PRESET.language, params,
                MVP_PRESET.model_size, created_at, created_at,
            ))

    def submit(self, source: InputSource) -> JobTicket:
        """Verify source, persist bytes, persist QUEUED, then return ticket.

        No SQLite entry is created on failed validation/copy. A handled SQLite
        failure removes only this unqueued job folder. A sudden process kill
        can leave an orphan folder but never an acknowledged QUEUED job.
        """
        if not isinstance(source, InputSource):
            raise TypeError("submit expects an InputSource")
        validated = validate_input(
            source, upload_roots=self.upload_roots, local_roots=self.local_roots,
            limits=self.limits, ffprobe_binary=self.ffprobe_binary,
        )
        job_id = str(uuid4())
        saved = self.storage.save_validated(
            job_id, validated, limits=self.limits, ffprobe_binary=self.ffprobe_binary,
        )
        created = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")
        try:
            self._insert_queued(saved, created)
        except Exception:
            self.storage.discard_unqueued(job_id)
            raise JobIntakeError(
                "QUEUE_PERSIST_FAILED", "Unable to persist the queued audio job"
            ) from None
        return JobTicket(
            job_id, "QUEUED", "QUEUED", saved.source.safe_filename,
            MVP_PRESET.model_size, MVP_PRESET.language,
            saved.source.duration_seconds, created,
        )

    def get(self, identifier: str) -> JobTicket | None:
        try:
            if str(UUID(identifier)) != identifier:
                return None
        except (ValueError, TypeError, AttributeError):
            return None
        with self._connect() as conn:
            row = conn.execute("""
                SELECT uuid, status, stage, file_name, model_name,
                       language, audio_duration, created_at
                FROM tasks WHERE uuid=?
            """, (identifier,)).fetchone()
        if row is None:
            return None
        return JobTicket(
            identifier=row["uuid"], status=row["status"], stage=row["stage"],
            file_name=row["file_name"], model_name=row["model_name"],
            language=row["language"], audio_duration=row["audio_duration"],
            created_at=row["created_at"],
        )

    def saved_input(self, identifier: str) -> Path:
        """Internal-only locator for P15 worker; never surface the path in UI."""
        ticket = self.get(identifier)
        if ticket is None:
            raise JobIntakeError("JOB_NOT_FOUND", "Job does not exist")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT input_path FROM tasks WHERE uuid=?", (identifier,)
            ).fetchone()
        if row is None or row["input_path"] is None:
            raise JobIntakeError("STORED_INPUT_MISSING", "Job audio is unavailable")
        return self.storage.resolve_input(row["input_path"])

"""P14 / GitHub PR15: submit and inspect durable queued transcription jobs.

No inference, worker, browser callback or separate FastAPI process is started.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Sequence
from uuid import UUID, uuid4

from backend.db.task.migrations import migrate_task_database
from modules.jobs.artifacts import ArtifactError, verify_result_set
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
    attempt: int = 0
    error_code: str | None = None
    progress: float = 0.0
    updated_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


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
        with closing(self._connect()) as conn:
            with conn:
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
            saved.source.duration_seconds, created, updated_at=created,
        )

    def get(self, identifier: str) -> JobTicket | None:
        try:
            if str(UUID(identifier)) != identifier:
                return None
        except (ValueError, TypeError, AttributeError):
            return None
        with closing(self._connect()) as conn:
            row = conn.execute("""
                SELECT uuid, status, stage, file_name, model_name,
                       language, audio_duration, created_at, attempt, error_code,
                       progress, updated_at, started_at, finished_at
                FROM tasks WHERE uuid=?
            """, (identifier,)).fetchone()
        if row is None:
            return None
        return JobTicket(
            identifier=row["uuid"], status=row["status"], stage=row["stage"],
            file_name=row["file_name"], model_name=row["model_name"],
            language=row["language"], audio_duration=row["audio_duration"],
            created_at=row["created_at"], attempt=int(row["attempt"]),
            error_code=row["error_code"],
            progress=float(row["progress"] or 0.0), updated_at=row["updated_at"],
            started_at=row["started_at"], finished_at=row["finished_at"],
        )

    def get_artifact(self, identifier: str, *, format: str) -> Path:
        """Resolve verified completed transcript for trusted local server code.

        Browser-supplied paths/names are never accepted; derive from UUID/DB.
        """
        if format not in ("json", "txt"):
            raise JobIntakeError("INVALID_ARTIFACT_FORMAT", "Choose json or txt")
        try:
            if str(UUID(identifier)) != identifier:
                raise ValueError("Noncanonical identifier")
        except (TypeError, ValueError, AttributeError):
            raise JobIntakeError("JOB_NOT_FOUND", "Job does not exist") from None
        with closing(self._connect()) as conn:
            row = conn.execute("""
                SELECT status, attempt, output_json_path, output_txt_path
                FROM tasks WHERE uuid=? AND task_type='TRANSCRIPTION'
            """, (identifier,)).fetchone()
        if row is None:
            raise JobIntakeError("JOB_NOT_FOUND", "Job does not exist")
        if row["status"] != "COMPLETED":
            raise JobIntakeError("ARTIFACT_NOT_READY", "Job has no completed result")
        attempt = int(row["attempt"])
        folder = self.storage.root / identifier / "result" / f"attempt-{attempt}"
        published = verify_result_set(folder, attempt=attempt)
        paths = {"json": published.files.json_path, "txt": published.files.txt_path}
        for label, column in (("json", "output_json_path"), ("txt", "output_txt_path")):
            if row[column] != paths[label].relative_to(self.storage.root).as_posix():
                raise ArtifactError("RESULT_CORRUPT", "Stored artifact locator disagrees")
        return paths[format]

    def get_status(self, identifier: str) -> JobTicket | None:
        """Retrieve a fresh SQLite status snapshot independent of browser sessions."""
        return self.get(identifier)

    def saved_input(self, identifier: str) -> Path:
        """Internal-only locator for P15 worker; never surface the path in UI."""
        ticket = self.get(identifier)
        if ticket is None:
            raise JobIntakeError("JOB_NOT_FOUND", "Job does not exist")
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT input_path FROM tasks WHERE uuid=?", (identifier,)
            ).fetchone()
        if row is None or row["input_path"] is None:
            raise JobIntakeError("STORED_INPUT_MISSING", "Job audio is unavailable")
        return self.storage.resolve_input(row["input_path"])


    def list_jobs(self, *, limit: int = 50) -> tuple[JobTicket, ...]:
        """Return recent jobs without exposing private input/output paths."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with closing(self._connect()) as conn:
            rows = conn.execute("""
                SELECT uuid, status, stage, file_name, model_name, language,
                       audio_duration, created_at, attempt, error_code,
                       progress, updated_at, started_at, finished_at
                FROM tasks WHERE task_type='TRANSCRIPTION'
                ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
        return tuple(
            JobTicket(
                identifier=row["uuid"], status=row["status"], stage=row["stage"],
                file_name=row["file_name"], model_name=row["model_name"],
                language=row["language"], audio_duration=row["audio_duration"],
                created_at=row["created_at"], attempt=int(row["attempt"]),
                error_code=row["error_code"], progress=float(row["progress"] or 0.0),
                updated_at=row["updated_at"], started_at=row["started_at"],
                finished_at=row["finished_at"],
            ) for row in rows
        )

    def cancel(self, identifier: str) -> bool:
        """P15: cancel only an unclaimed queued job; never kill active inference.

        P19 adds cooperative cancellation of an already running transcription.
        """
        if self.get(identifier) is None:
            return False
        at = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")
        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute("""
                    UPDATE tasks SET status='CANCELLED', stage='CANCELLED',
                        finished_at=?, updated_at=?
                    WHERE uuid=? AND status='QUEUED'
                """, (at, at, identifier))
                return cursor.rowcount == 1

    def retry(self, identifier: str) -> bool:
        """Explicitly requeue FAILED or recovered INTERRUPTED with saved input.

        Never automatically restart interrupted inference. The next claim
        increments attempt; earlier published artifacts remain unchanged.
        """
        ticket = self.get(identifier)
        if ticket is None or ticket.status not in ("FAILED", "INTERRUPTED"):
            return False
        # Explicitly check durable input before changing the queue state.
        try:
            self.saved_input(identifier)
        except (JobIntakeError, JobStorageError):
            return False
        at = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")
        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute("""
                    UPDATE tasks SET status='QUEUED', stage='QUEUED',
                        error=NULL, error_code=NULL, started_at=NULL,
                        finished_at=NULL, output_json_path=NULL,
                        output_txt_path=NULL, progress=0.0, updated_at=?
                    WHERE uuid=? AND status IN ('FAILED', 'INTERRUPTED')
                """, (at, identifier))
                return cursor.rowcount == 1

"""PR16 / P15: exactly one local FIFO inference worker, independent of Gradio.

The durable SQLite queue is authoritative. No job executes in a request thread.
Legacy Gradio callbacks must not run ASR concurrently with this worker.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import inspect
import os
from pathlib import Path
import re
import sqlite3
from threading import Event, Lock, Thread, current_thread
from typing import Callable

from modules.jobs.inputs import InputSource, InputValidationError, validate_input
from modules.jobs.results import ResultFormatError
from modules.jobs.artifacts import ArtifactError, publish_result_set
from modules.jobs.progress import progress_for_stage, stage_index
from modules.jobs.service import JobService
from modules.jobs.storage import JobStorageError
from modules.jobs.transcription_service import (
    TranscriptionService, TranscriptionServiceError,
)

logger = logging.getLogger(__name__)


class WorkerError(RuntimeError):
    """Non-sensitive error code suitable for display, with no audio data."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")


class _ExclusiveWorkerLock:
    """Cross-process single owner on Windows (msvcrt) and Unix (flock).

    Do not remove the lock file: the descriptor, not file existence, owns it.
    """

    def __init__(self, path: Path):
        self._path = path
        self._handle = None

    def acquire(self) -> None:
        if self._handle is not None:
            raise WorkerError("WORKER_BUSY", "Worker already owns the queue")
        if self._path.is_symlink():
            raise WorkerError("UNSAFE_WORKER_LOCK", "Worker lock path is unsafe")
        try:
            handle = self._path.open("a+b")
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError):
                handle.close()
                raise WorkerError(
                    "WORKER_BUSY", "Another job worker already owns this directory"
                ) from None
            self._handle = handle
        except WorkerError:
            raise
        except OSError:
            raise WorkerError("WORKER_LOCK_FAILED", "Could not lock job storage") from None

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@dataclass(frozen=True, slots=True)
class _Claimed:
    identifier: str
    attempt: int


class JobWorker:
    """Sequential worker. Model instance is created lazily inside its thread.

    Starting the worker does NOT alter legacy app.py or create a second server.
    A single exclusive lock guards the complete worker lifetime, including ASR.
    """

    def __init__(
        self, jobs: JobService, *,
        transcriber_factory: Callable[[], object] = TranscriptionService,
        poll_interval: float = 0.5,
    ):
        if not isinstance(jobs, JobService):
            raise TypeError("Expected the durable JobService from PR15")
        if not callable(transcriber_factory):
            raise TypeError("transcriber_factory must be callable")
        if not 0.01 <= poll_interval <= 60:
            raise ValueError("poll_interval must be between 0.01 and 60 seconds")
        self.jobs = jobs
        self._factory = transcriber_factory
        self._poll_interval = poll_interval
        self._lock = _ExclusiveWorkerLock(jobs.storage.root / ".worker.lock")
        self._lifecycle = Lock()
        self._stop = Event()
        self._wake = Event()
        self._thread: Thread | None = None
        self._failure_code: str | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def failure_code(self) -> str | None:
        return self._failure_code

    def start(self) -> None:
        """Take exclusive ownership and refuse stale IN_PROGRESS jobs.

        P18 will implement safe restart reconciliation; never silently resume a
        crashed inference or launch new work alongside a stale claim.
        """
        with self._lifecycle:
            if self.is_running:
                raise WorkerError("WORKER_BUSY", "Job worker is already running")
            self._lock.acquire()
            try:
                with closing(self.jobs._connect()) as conn:
                    running = conn.execute(
                        "SELECT COUNT(*) FROM tasks WHERE status='IN_PROGRESS' "
                        "AND task_type='TRANSCRIPTION'"
                    ).fetchone()[0]
                if running:
                    raise WorkerError(
                        "RECOVERY_REQUIRED",
                        "Interrupted jobs require recovery before starting the worker",
                    )
                self._stop.clear()
                self._wake.clear()
                self._failure_code = None
                self._thread = Thread(
                    target=self._run, name="WhisperX-single-job-worker", daemon=True,
                )
                self._thread.start()
            except Exception:
                self._lock.release()
                raise

    def wake(self) -> None:
        """Optional notice after submit; polling also finds persisted jobs."""
        self._wake.set()

    def stop(self, *, wait: bool = True, timeout: float | None = None) -> bool:
        """Stop claiming new tasks; current inference is not force-aborted.

        A non-waiting stop leaves the exclusive lock held until the model exits.
        Returns False if the worker still runs after a requested timeout.
        """
        with self._lifecycle:
            thread = self._thread
            self._stop.set()
            self._wake.set()
        if thread is None:
            return True
        if wait:
            if current_thread() is thread:
                raise WorkerError("INVALID_STOP", "Worker cannot join itself")
            thread.join(timeout)
        return not thread.is_alive()

    def _claim(self) -> _Claimed | None:
        with closing(self.jobs._connect()) as conn:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                stale = conn.execute(
                    "SELECT 1 FROM tasks WHERE status='IN_PROGRESS' "
                    "AND task_type='TRANSCRIPTION' LIMIT 1"
                ).fetchone()
                if stale:
                    raise WorkerError(
                        "RECOVERY_REQUIRED",
                        "Other in-progress job requires explicit recovery",
                    )
                job = conn.execute("""
                    SELECT id, uuid, attempt FROM tasks
                    WHERE status='QUEUED' AND task_type='TRANSCRIPTION'
                    ORDER BY id ASC LIMIT 1
                """).fetchone()
                if job is None:
                    return None
                start = _now()
                cursor = conn.execute("""
                    UPDATE tasks SET status='IN_PROGRESS', stage='PREPARING', progress=0.0,
                        attempt=attempt+1, started_at=?, updated_at=?,
                        error=NULL, error_code=NULL
                    WHERE id=? AND status='QUEUED'
                """, (start, start, job["id"]))
                if cursor.rowcount != 1:
                    raise WorkerError("QUEUE_CONFLICT", "Unable to claim queued job")
                return _Claimed(job["uuid"], int(job["attempt"]) + 1)

    def _finish(
        self, claim: _Claimed, *, success: bool, error_code: str | None = None,
        txt: str | None = None, json_file: str | None = None,
    ) -> None:
        final = "COMPLETED" if success else "FAILED"
        phase = "DONE" if success else "FAILED"
        message = None if success else "Audio processing failed; see error_code"
        with closing(self.jobs._connect()) as conn:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                updated = conn.execute("""
                    UPDATE tasks SET status=?, stage=?, error_code=?, error=?,
                        output_txt_path=?, output_json_path=?,
                        progress=CASE WHEN ?='COMPLETED' THEN 1.0 ELSE progress END,
                        finished_at=?, updated_at=?
                    WHERE uuid=? AND status='IN_PROGRESS' AND attempt=?
                """, (
                    final, phase, error_code, message, txt, json_file,
                    final, _now(), _now(),
                    claim.identifier, claim.attempt,
                ))
                if updated.rowcount != 1:
                    raise WorkerError("JOB_STATE_CONFLICT", "Job status changed unexpectedly")

    def _record_stage(
        self, claim: _Claimed, stage: str, *, asr_fraction: float | None = None,
    ) -> bool:
        """Persist monotone actual milestones only for the current job attempt."""
        value = progress_for_stage(stage, asr_fraction=asr_fraction)
        rank = stage_index(stage)
        with closing(self.jobs._connect()) as conn:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("""
                    SELECT stage, progress FROM tasks
                    WHERE uuid=? AND attempt=? AND status='IN_PROGRESS'
                """, (claim.identifier, claim.attempt)).fetchone()
                if row is None:
                    return False
                try:
                    previous_rank = stage_index(row["stage"])
                except ValueError:
                    return False
                previous = float(row["progress"] or 0.0)
                if rank < previous_rank or value <= previous:
                    return False
                at = _now()
                updated = conn.execute("""
                    UPDATE tasks SET stage=?, progress=?, updated_at=?
                    WHERE uuid=? AND attempt=? AND status='IN_PROGRESS'
                """, (stage, value, at, claim.identifier, claim.attempt))
                return updated.rowcount == 1

    def _callbacks_for(self, transcriber: object, claim: _Claimed) -> dict:
        """Use real service events; retain older fake/custom adapters unchanged."""
        try:
            params = inspect.signature(transcriber.transcribe).parameters
        except (TypeError, ValueError):
            return {}
        supports_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs: dict = {}
        if "stage_callback" in params or supports_kw:
            kwargs["stage_callback"] = lambda stage: self._record_stage(claim, stage)
        if "progress_callback" in params or supports_kw:
            def on_asr(value: float) -> None:
                try:
                    self._record_stage(claim, "ASR", asr_fraction=value)
                except ValueError:
                    # An invalid model progress fraction is not an ASR failure.
                    return
            kwargs["progress_callback"] = on_asr
        return kwargs

    def _process(self, claim: _Claimed, transcriber: object) -> None:
        try:
            path = self.jobs.saved_input(claim.identifier)
            # Fresh validation of the stored copy: never trust a vanished cache.
            source = validate_input(
                InputSource.local_file(path),
                local_roots=(path.parent,),
                limits=self.jobs.limits,
                ffprobe_binary=self.jobs.ffprobe_binary,
            )
            ticket = self.jobs.get(claim.identifier)
            if ticket is None:
                raise WorkerError("JOB_NOT_FOUND", "Accepted job was not found")
            from dataclasses import replace
            source = replace(source, safe_filename=ticket.file_name)
            result = transcriber.transcribe(source, **self._callbacks_for(transcriber, claim))
            self._record_stage(claim, "EXPORT")
            job_dir = self.jobs.storage.root / claim.identifier
            # Atomically reveal JSON, TXT and their checksum manifest together.
            # No COMPLETED transition until the full bundle has been verified.
            published = publish_result_set(result, job_dir, attempt=claim.attempt)
            self._finish(
                claim, success=True,
                json_file=str(published.files.json_path.relative_to(self.jobs.storage.root)),
                txt=str(published.files.txt_path.relative_to(self.jobs.storage.root)),
            )
        except Exception as exc:
            accepted = (
                TranscriptionServiceError, InputValidationError,
                JobStorageError, ResultFormatError, ArtifactError, WorkerError,
            )
            code = getattr(exc, "code", None) if isinstance(exc, accepted) else None
            if not isinstance(code, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", code):
                code = "PROCESSING_FAILED"
            # Never persist arbitrary exception strings (paths/tokens/audio).
            try:
                self._finish(claim, success=False, error_code=code)
            except Exception:
                logger.error("Worker could not persist FAILED; recovery required")
                raise

    def _run(self) -> None:
        transcriber: object | None = None
        try:
            while not self._stop.is_set():
                claim = self._claim()
                if claim is None:
                    self._wake.wait(self._poll_interval)
                    self._wake.clear()
                    continue
                if transcriber is None:
                    try:
                        transcriber = self._factory()
                    except Exception:
                        self._finish(
                            claim, success=False, error_code="MODEL_INIT_FAILED"
                        )
                        continue
                self._process(claim, transcriber)
        except Exception:
            self._failure_code = "WORKER_STOPPED"
            logger.error("Single job worker stopped; recovery required")
        finally:
            self._lock.release()

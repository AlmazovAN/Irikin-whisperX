"""PR19 / P18: crash recovery for a *locked*, single-worker job directory.

Executed only by JobWorker.start() after acquiring its cross-process lock.
No inference, retry, deletion, or external-network action occurs here.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from uuid import UUID

from modules.jobs.artifacts import ArtifactError, verify_result_set
from modules.jobs.service import JobService

_PENDING_NAME = re.compile(r"\.attempt-[1-9][0-9]*-pending-[0-9a-f]{32}\Z")


class RecoveryError(RuntimeError):
    """Non-sensitive recovery diagnostics with stable error codes."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """Counts only. Never reveal private filenames or transcription text."""

    completed_from_artifacts: int = 0
    interrupted: int = 0
    unverified_outputs: int = 0
    queued_preserved: int = 0
    orphan_job_directories: int = 0
    unfinished_output_directories: int = 0


def _now() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")


def _valid_uuid(identifier: object) -> bool:
    try:
        return isinstance(identifier, str) and str(UUID(identifier)) == identifier
    except (TypeError, ValueError, AttributeError):
        return False


def _inspect_disk(root: Path, known: set[str]) -> tuple[int, int]:
    """Read-only scan of strictly UUID-named children; never follow symlinks.

    Temp outputs remain available for manual review or later cleanup policy.
    """
    orphans = pending = 0
    try:
        children = list(root.iterdir())
    except OSError:
        raise RecoveryError("RECOVERY_STORAGE_FAILED", "Cannot inspect job storage") from None
    for job_dir in children:
        if not _valid_uuid(job_dir.name):
            continue
        if job_dir.name not in known:
            orphans += 1
        if job_dir.is_symlink() or not job_dir.is_dir():
            continue
        outputs = job_dir / "result"
        if outputs.is_symlink() or not outputs.is_dir():
            continue
        try:
            pending += sum(
                1 for child in outputs.iterdir()
                if _PENDING_NAME.fullmatch(child.name)
            )
        except OSError:
            raise RecoveryError(
                "RECOVERY_STORAGE_FAILED", "Cannot inspect unfinished outputs"
            ) from None
    return orphans, pending


def _recover_jobs_under_lock(jobs: JobService) -> RecoveryReport:
    """Reconcile persisted IN_PROGRESS only while the caller owns worker lock.

    A valid published *current-attempt* result is committed as COMPLETED without
    rerunning inference. Every other IN_PROGRESS becomes INTERRUPTED, retaining
    the input and any partial artifacts for explicit operator retry.

    QUEUED, FAILED, COMPLETED and CANCELLED are never rewritten. If the SQLite
    transaction fails, no partial DB reconciliation is committed.
    """
    if not isinstance(jobs, JobService):
        raise TypeError("Recovery requires an existing JobService")
    root = jobs.storage.root
    with closing(jobs._connect()) as conn:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute("""
                SELECT uuid, status, attempt, error_code
                FROM tasks WHERE task_type='TRANSCRIPTION' ORDER BY id
            """).fetchall()
            known = {row["uuid"] for row in rows}
            orphan_count, temp_count = _inspect_disk(root, known)
            completed = interrupted = corrupt = queued = 0
            for row in rows:
                status = row["status"]
                if status == "QUEUED":
                    queued += 1
                    continue
                if status != "IN_PROGRESS":
                    continue
                identifier, attempt = row["uuid"], row["attempt"]
                if row["error_code"] == "CANCEL_REQUESTED":
                    # The cancel intent is committed before this worker died.
                    # Never resurrect an unwanted artifact as COMPLETED.
                    at = _now()
                    changed = conn.execute("""
                        UPDATE tasks SET status='CANCELLED', stage='CANCELLED',
                            error_code=NULL, error=NULL,
                            output_json_path=NULL, output_txt_path=NULL,
                            finished_at=?, updated_at=?
                        WHERE uuid=? AND status='IN_PROGRESS' AND attempt=?
                          AND error_code='CANCEL_REQUESTED'
                    """, (at, at, identifier, attempt))
                    if changed.rowcount != 1:
                        raise RecoveryError("RECOVERY_CONFLICT", "Job changed during recovery")
                    continue
                verified = None
                damaged = False
                if _valid_uuid(identifier) and type(attempt) is int and attempt >= 1:
                    job_dir = root / identifier
                    result_dir = job_dir / "result"
                    candidate = result_dir / f"attempt-{attempt}"
                    # Refuse symlinked parents, even if content passes checksum.
                    if job_dir.is_symlink() or result_dir.is_symlink():
                        damaged = True
                    elif candidate.exists() or candidate.is_symlink():
                        try:
                            verified = verify_result_set(candidate, attempt=attempt)
                        except (ArtifactError, OSError, ValueError):
                            damaged = True
                elif not _valid_uuid(identifier):
                    damaged = True

                at = _now()
                if verified is not None:
                    json_rel = verified.files.json_path.relative_to(root).as_posix()
                    txt_rel = verified.files.txt_path.relative_to(root).as_posix()
                    changed = conn.execute("""
                        UPDATE tasks SET status='COMPLETED', stage='DONE',
                            progress=1.0, error=NULL, error_code=NULL,
                            output_json_path=?, output_txt_path=?,
                            finished_at=?, updated_at=?
                        WHERE uuid=? AND status='IN_PROGRESS' AND attempt=?
                    """, (json_rel, txt_rel, at, at, identifier, attempt))
                    completed += 1
                else:
                    code = "RECOVERY_OUTPUT_UNVERIFIED" if damaged else "RECOVERY_INTERRUPTED"
                    changed = conn.execute("""
                        UPDATE tasks SET status='INTERRUPTED', stage='INTERRUPTED',
                            error_code=?, error='Job interrupted by server shutdown',
                            output_json_path=NULL, output_txt_path=NULL,
                            finished_at=?, updated_at=?
                        WHERE uuid=? AND status='IN_PROGRESS' AND attempt=?
                    """, (code, at, at, identifier, attempt))
                    interrupted += 1
                    if damaged:
                        corrupt += 1
                if changed.rowcount != 1:
                    raise RecoveryError("RECOVERY_CONFLICT", "Job changed during recovery")
    return RecoveryReport(
        completed_from_artifacts=completed, interrupted=interrupted,
        unverified_outputs=corrupt, queued_preserved=queued,
        orphan_job_directories=orphan_count,
        unfinished_output_directories=temp_count,
    )

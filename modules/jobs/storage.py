"""P14 / GitHub PR15: durable, locally owned audio intake.

The caller supplies a P08 ValidatedInput; this code NEVER starts WhisperX.
Input is published by atomic rename before the SQLite queue record is committed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import shutil
import stat
from uuid import UUID, uuid4

from modules.jobs.inputs import (
    InputKind, InputLimits, InputSource, InputValidationError, ValidatedInput,
    _safe_filename, validate_input,
)

DEFAULT_JOB_ROOT = Path(__file__).resolve().parents[2] / "data" / "jobs"
_CHUNK_BYTES = 1024 * 1024


class JobStorageError(RuntimeError):
    """Stable error code; messages are safe for UI and omit filesystem paths."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class StoredAudio:
    job_id: str
    relative_path: str
    source: ValidatedInput
    sha256: str


def _canonical_job_id(job_id: str) -> str:
    try:
        normalized = str(UUID(job_id))
    except (ValueError, TypeError, AttributeError):
        raise JobStorageError("INVALID_JOB_ID", "Invalid job identifier") from None
    if normalized != job_id or len(job_id) != 36:
        raise JobStorageError("INVALID_JOB_ID", "Invalid job identifier")
    return normalized


def _copy_verified(source: ValidatedInput, part: Path, limit: int) -> str:
    """Open source once, reject symlinks and changed file metadata; fsync output."""
    try:
        path = source.path
        if path.resolve(strict=True) != path or path.is_symlink():
            raise JobStorageError("INPUT_CHANGED", "Input changed after validation")
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_size != source.size_bytes:
            raise JobStorageError("INPUT_CHANGED", "Input changed after validation")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as src:
            opened = os.fstat(src.fileno())
            if not stat.S_ISREG(opened.st_mode) or (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns
            ) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
                raise JobStorageError("INPUT_CHANGED", "Input changed after validation")
            target_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            dest_fd = os.open(part, target_flags, 0o600)
            digest = hashlib.sha256()
            copied = 0
            with os.fdopen(dest_fd, "wb") as dst:
                while True:
                    chunk = src.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > limit or copied > source.size_bytes:
                        raise JobStorageError("INPUT_CHANGED", "Audio exceeds validated size")
                    dst.write(chunk)
                    digest.update(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            after = os.fstat(src.fileno())
            if copied != source.size_bytes or (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
            ) != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
                raise JobStorageError("INPUT_CHANGED", "Input changed during copying")
            if path.is_symlink() or path.stat().st_ino != opened.st_ino:
                raise JobStorageError("INPUT_CHANGED", "Input changed during copying")
        return digest.hexdigest()
    except JobStorageError:
        raise
    except (OSError, ValueError, RuntimeError):
        raise JobStorageError("INPUT_UNAVAILABLE", "Unable to safely copy audio") from None


class JobStorage:
    """One durable, trusted local root. All job directories are UUID-addressed."""

    def __init__(self, root: str | Path = DEFAULT_JOB_ROOT):
        candidate = Path(root).expanduser()
        if candidate.is_symlink():
            raise JobStorageError("UNSAFE_STORAGE_ROOT", "Job storage cannot be a symlink")
        candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = candidate.resolve(strict=True)
        if not self.root.is_dir():
            raise JobStorageError("UNSAFE_STORAGE_ROOT", "Invalid job storage directory")

    @property
    def database_path(self) -> Path:
        return self.root / "jobs.sqlite3"

    def save_validated(
        self, job_id: str, source: ValidatedInput, *,
        limits: InputLimits, ffprobe_binary: str = "ffprobe",
    ) -> StoredAudio:
        """Create an exclusive input folder, fsync/copy/publish, then re-probe.

        On a known error, remove only the new UUID directory. A process crash
        between rename and DB commit can leave an orphan; P18 handles recovery.
        """
        job_id = _canonical_job_id(job_id)
        if not isinstance(source, ValidatedInput) or not isinstance(limits, InputLimits):
            raise TypeError("Expected validated audio and input limits")
        if source.source_kind not in (InputKind.UPLOAD, InputKind.LOCAL_FILE):
            raise JobStorageError("UNSUPPORTED_SOURCE", "Unsupported local audio source")
        name = _safe_filename(source.safe_filename)
        file_name = f"{uuid4().hex}-{name}"
        job_dir = self.root / job_id
        try:
            job_dir.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError:
            raise JobStorageError("JOB_EXISTS", "Job identifier already exists") from None
        input_dir = job_dir / "input"
        part = input_dir / f".{file_name}.part"
        final = input_dir / file_name
        try:
            input_dir.mkdir(mode=0o700)
            digest = _copy_verified(source, part, limits.max_file_bytes)
            os.replace(part, final)
            # Recheck the bytes actually owned by the job, not the Gradio cache.
            checked = validate_input(
                InputSource.local_file(final), local_roots=(input_dir,),
                limits=limits, ffprobe_binary=ffprobe_binary,
            )
            if checked.size_bytes != source.size_bytes or abs(
                checked.duration_seconds - source.duration_seconds
            ) > max(0.25, source.duration_seconds * 0.01):
                raise JobStorageError("INPUT_CHANGED", "Stored audio does not match its validation")
            canonical = final.resolve(strict=True)
            stored = replace(
                checked, source_kind=source.source_kind, path=canonical,
                safe_filename=name, storage_filename=file_name,
            )
            # P13 Task.input_path is private: store only a root-relative locator.
            return StoredAudio(job_id, f"{job_id}/input/{file_name}", stored, digest)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

    def discard_unqueued(self, job_id: str) -> None:
        """Called only by the submitting owner after a failed DB transaction."""
        job_id = _canonical_job_id(job_id)
        shutil.rmtree(self.root / job_id, ignore_errors=True)

    def resolve_input(self, relative_path: str) -> Path:
        """Never accept an arbitrary absolute/path-traversal path from a caller."""
        parts = Path(relative_path).parts
        if len(parts) != 3 or parts[1] != "input":
            raise JobStorageError("INVALID_INPUT_PATH", "Unexpected stored input locator")
        _canonical_job_id(parts[0])
        target = self.root.joinpath(*parts)
        try:
            resolved = target.resolve(strict=True)
        except (OSError, RuntimeError):
            raise JobStorageError("STORED_INPUT_MISSING", "Stored input is unavailable") from None
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise JobStorageError("INVALID_INPUT_PATH", "Stored input location is invalid")
        return resolved

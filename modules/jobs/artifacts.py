"""PR18 / P17: crash-conscious publication of complete transcription artifacts.

Publish a JSON/TXT/manifest *directory* in one same-filesystem rename.
A published result set is immutable and bound to a particular job attempt.
No automatic deletion of user audio or transcripts is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from uuid import uuid4

from modules.jobs.results import (
    ResultFiles, build_result_payload, render_json, render_txt,
)

MANIFEST_VERSION = 1
_NAMES = ("transcript.json", "transcript.txt")
_MAX_MANIFEST_BYTES = 4096
_CHUNK_SIZE = 1024 * 1024


class ArtifactError(ValueError):
    """Failure codes never contain file paths, audio text, or secrets."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PublishedResult:
    files: ResultFiles
    manifest_path: Path
    attempt: int


def _valid_attempt(attempt: int) -> None:
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ArtifactError("INVALID_ATTEMPT", "A positive attempt number is required")


def _regular_directory(folder: Path) -> None:
    # Refuse a replaced job directory or symlinked result parent as well as a
    # symlinked published attempt. Do not follow paths outside durable jobs.
    if (folder.is_symlink() or folder.parent.is_symlink()
            or folder.parent.parent.is_symlink() or not folder.is_dir()):
        raise ArtifactError("UNSAFE_RESULT_DIR", "Result directory is missing or unsafe")


def _fsync_directory(folder: Path) -> None:
    # POSIX supports syncing directory entries; Windows gives no equivalent
    # portable stdlib primitive. We still fsync every file on both systems.
    if os.name == "posix":
        fd = os.open(folder, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _write_fsynced(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _read_digest(path: Path) -> tuple[int, str]:
    if path.is_symlink():
        raise ArtifactError("RESULT_CORRUPT", "Published artifact is unsafe")
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactError("RESULT_CORRUPT", "Published artifact is not a file")
        total = 0
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise ArtifactError("RESULT_CORRUPT", "Artifact changed during reading")
            while chunk := stream.read(_CHUNK_SIZE):
                total += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        if (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns):
            raise ArtifactError("RESULT_CORRUPT", "Artifact changed during reading")
        return total, digest.hexdigest()
    except ArtifactError:
        raise
    except OSError:
        raise ArtifactError("RESULT_CORRUPT", "Published artifact could not be read") from None


def verify_result_set(directory: str | Path, *, attempt: int) -> PublishedResult:
    """Verify complete immutable set and SHA-256 digests before use.

    Call with an internally derived per-job directory, not a browser path.
    """
    _valid_attempt(attempt)
    folder = Path(directory)
    _regular_directory(folder)
    if folder.name != f"attempt-{attempt}":
        raise ArtifactError("RESULT_CORRUPT", "Unexpected artifact attempt directory")
    manifest_path = folder / "manifest.json"
    if manifest_path.is_symlink():
        raise ArtifactError("RESULT_CORRUPT", "Published manifest is unsafe")
    try:
        if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise ValueError("Manifest too large")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict)
                or manifest.get("version") != MANIFEST_VERSION
                or type(manifest.get("attempt")) is not int
                or manifest["attempt"] != attempt
                or not isinstance(manifest.get("files"), dict)
                or set(manifest["files"]) != set(_NAMES)):
            raise ValueError("Invalid manifest")
        for name in _NAMES:
            meta = manifest["files"][name]
            if (not isinstance(meta, dict)
                    or type(meta.get("bytes")) is not int
                    or meta["bytes"] < 1
                    or not isinstance(meta.get("sha256"), str)
                    or len(meta["sha256"]) != 64):
                raise ValueError("Invalid artifact digest")
            count, digest = _read_digest(folder / name)
            if count != meta["bytes"] or digest != meta["sha256"]:
                raise ValueError("Result checksum mismatch")
        # Parse JSON for structural validity in addition to a valid checksum.
        payload = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
            raise ValueError("Invalid transcript JSON")
    except (OSError, ValueError, UnicodeError, TypeError, KeyError):
        raise ArtifactError("RESULT_CORRUPT", "Published result failed integrity validation") from None
    return PublishedResult(
        files=ResultFiles(folder / "transcript.json", folder / "transcript.txt"),
        manifest_path=manifest_path, attempt=attempt,
    )


def publish_result_set(
    result: object, job_directory: str | Path, *, attempt: int,
) -> PublishedResult:
    """Write both outputs and manifest, then publish as a single directory.

    A known pre-publication failure removes only the temporary directory.
    An already published attempt is never replaced or deleted.
    """
    _valid_attempt(attempt)
    job_dir = Path(job_directory)
    _regular_directory(job_dir)
    outputs = job_dir / "result"
    if outputs.is_symlink():
        raise ArtifactError("UNSAFE_RESULT_DIR", "Result directory is unsafe")
    try:
        outputs.mkdir(mode=0o700, exist_ok=True)
    except OSError:
        raise ArtifactError("RESULT_WRITE_FAILED", "Unable to create output directory") from None
    _regular_directory(outputs)
    final = outputs / f"attempt-{attempt}"
    if final.exists() or final.is_symlink():
        raise ArtifactError("RESULT_EXISTS", "The result attempt already exists")

    # Format and validate before creating files; do not modify raw Segments.
    payload = build_result_payload(result)
    serialized = {
        "transcript.json": render_json(payload).encode("utf-8"),
        "transcript.txt": render_txt(payload).encode("utf-8"),
    }
    stage = outputs / f".attempt-{attempt}-pending-{uuid4().hex}"
    published = False
    try:
        stage.mkdir(mode=0o700)
        checksums = {}
        for name, data in serialized.items():
            _write_fsynced(stage / name, data)
            checksums[name] = {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        manifest = {
            "version": MANIFEST_VERSION,
            "attempt": attempt,
            "files": checksums,
        }
        _write_fsynced(
            stage / "manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        _fsync_directory(stage)
        # Within the same filesystem the directory rename exposes all files
        # together. Single worker ownership prevents a competing publisher.
        if final.exists() or final.is_symlink():
            raise ArtifactError("RESULT_EXISTS", "The result attempt already exists")
        stage.rename(final)
        published = True
        _fsync_directory(outputs)
        return verify_result_set(final, attempt=attempt)
    except ArtifactError:
        raise
    except (OSError, ValueError):
        raise ArtifactError("RESULT_WRITE_FAILED", "Cannot publish transcription results") from None
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

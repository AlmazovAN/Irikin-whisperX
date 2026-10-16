"""P08: validate local audio inputs before handing files to WhisperX.

This module is intentionally stdlib-only: validating an upload cannot load
PyTorch, audio decoders, an inference model, or persist a job.  Both trusted
server-side upload directories and approved local file directories are explicit.
The next phase (P14) will copy validated content into durable job storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import unicodedata
from typing import Sequence
from uuid import uuid4

AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".ogg"})
_WINDOWS_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL"}) | {
    f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)
}


class InputKind(str, Enum):
    UPLOAD = "upload"
    LOCAL_FILE = "local_file"
    GOOGLE_DRIVE_URL = "google_drive_url"  # Contract only; handled in P26+.


class InputValidationError(ValueError):
    """Predictable, path-free validation failure safe to surface to the UI."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class InputSource:
    kind: InputKind
    location: str
    original_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, InputKind):
            raise InputValidationError("INVALID_SOURCE", "Unknown input source type")
        if not isinstance(self.location, str) or not self.location.strip() or "\x00" in self.location:
            raise InputValidationError("INVALID_SOURCE", "Input location must be a nonempty string")
        if self.original_name is not None and not isinstance(self.original_name, str):
            raise InputValidationError("INVALID_NAME", "Original filename must be text")

    @classmethod
    def upload(cls, file_path: str | Path, *, original_name: str | None = None) -> InputSource:
        return cls(InputKind.UPLOAD, str(file_path), original_name)

    @classmethod
    def local_file(cls, file_path: str | Path) -> InputSource:
        return cls(InputKind.LOCAL_FILE, str(file_path))

    @classmethod
    def google_drive_url(cls, url: str) -> InputSource:
        return cls(InputKind.GOOGLE_DRIVE_URL, url)


@dataclass(frozen=True, slots=True)
class InputLimits:
    max_file_bytes: int = 512 * 1024 * 1024
    max_duration_seconds: float = 4 * 3600
    ffprobe_timeout_seconds: float = 15

    def __post_init__(self) -> None:
        if (type(self.max_file_bytes) is not int or self.max_file_bytes < 1
                or not math.isfinite(self.max_duration_seconds)
                or self.max_duration_seconds <= 0
                or not math.isfinite(self.ffprobe_timeout_seconds)
                or self.ffprobe_timeout_seconds <= 0):
            raise ValueError("Input limits must be positive, finite numbers")


@dataclass(frozen=True, slots=True)
class ValidatedInput:
    source_kind: InputKind
    path: Path  # Only the server uses this; never echo to client/logs.
    safe_filename: str
    storage_filename: str  # Unique basename only; P14 chooses the storage directory.
    size_bytes: int
    duration_seconds: float
    codec_name: str


def _safe_filename(untrusted: str) -> str:
    """Preserve readable Unicode while stripping traversal and Windows metacharacters."""
    basename = unicodedata.normalize("NFKC", untrusted.replace("\\", "/").rsplit("/", 1)[-1])
    suffix = Path(basename).suffix.lower()
    if suffix not in AUDIO_EXTENSIONS:
        raise InputValidationError("UNSUPPORTED_FORMAT", "Only MP3, WAV, M4A and OGG are supported")
    stem = basename[: -len(suffix)]
    cleaned = "".join(ch if (ch.isalnum() or ch in " ._-") else "_" for ch in stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._")[:100].rstrip(" ._") or "audio"
    if cleaned.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        cleaned = "audio_" + cleaned
    return cleaned + suffix


def _allowed_path(raw_path: str, roots: Sequence[str | Path]) -> Path:
    if not roots:
        raise InputValidationError("ACCESS_DENIED", "No input directory has been approved")
    try:
        candidate = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise InputValidationError("FILE_NOT_FOUND", "Input file is unavailable") from None
    if not candidate.is_file():
        raise InputValidationError("NOT_A_FILE", "Input must be a regular file")
    for raw_root in roots:
        try:
            root = Path(raw_root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            continue
        if not root.is_dir() or root == Path(root.anchor):
            continue  # Reject entire drive/filesystem as a grant.
        if candidate.is_relative_to(root):
            return candidate
    raise InputValidationError("ACCESS_DENIED", "File is outside approved input directories")


def _probe_audio(path: Path, *, binary: str, timeout: float) -> tuple[float, str]:
    executable = shutil.which(binary)
    if not executable:
        raise InputValidationError("FFPROBE_UNAVAILABLE", "FFprobe is not installed or not on PATH")
    try:
        completed = subprocess.run(
            [executable, "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,codec_name,duration:stream_disposition=attached_pic",
             "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, stdin=subprocess.DEVNULL, check=False,
        )
    except subprocess.TimeoutExpired:
        raise InputValidationError("FFPROBE_TIMEOUT", "Audio inspection exceeded the time limit") from None
    except OSError:
        raise InputValidationError("FFPROBE_FAILED", "Unable to start FFprobe") from None
    if completed.returncode != 0:
        raise InputValidationError("INVALID_MEDIA", "FFprobe could not read the audio file")
    try:
        info = json.loads(completed.stdout)
        streams = info.get("streams", [])
        if not isinstance(streams, list):
            raise ValueError("Invalid streams")
        if any(s.get("codec_type") == "video" and not
               s.get("disposition", {}).get("attached_pic") for s in streams):
            raise ValueError("Video content is not supported")
        audio_streams = [s for s in streams if s.get("codec_type") == "audio" and s.get("codec_name")]
        if not audio_streams:
            raise ValueError("No audio stream")
        candidates = [info.get("format", {}).get("duration")]
        candidates += [s.get("duration") for s in audio_streams]
        durations = [float(v) for v in candidates if v is not None and str(v).strip()]
        durations = [d for d in durations if math.isfinite(d) and d > 0]
        if not durations:
            raise ValueError("Missing positive duration")
        return max(durations), str(audio_streams[0]["codec_name"])
    except (TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError):
        raise InputValidationError("INVALID_MEDIA", "Audio metadata is missing or invalid") from None


def validate_input(
    source: InputSource,
    *,
    upload_roots: Sequence[str | Path] = (),
    local_roots: Sequence[str | Path] = (),
    limits: InputLimits | None = None,
    ffprobe_binary: str = "ffprobe",
) -> ValidatedInput:
    """Authorize a file, inspect it, and return an immutable intake descriptor.

    Roots are provided by trusted server configuration, never by browser input.
    An upload must be in a Gradio staging root; a local file must be in an
    explicitly approved local root. Paths are resolved to block symlink escapes.
    This validates metadata only; P14 must recheck when copying from staging.
    """
    if not isinstance(source, InputSource):
        raise InputValidationError("INVALID_SOURCE", "Expected InputSource")
    if source.kind is InputKind.GOOGLE_DRIVE_URL:
        raise InputValidationError("SOURCE_NOT_SUPPORTED", "Google Drive import is planned for a later release")
    current_limits = limits if limits is not None else InputLimits()
    if not isinstance(current_limits, InputLimits):
        raise TypeError("limits must be InputLimits")
    roots = upload_roots if source.kind is InputKind.UPLOAD else local_roots
    path = _allowed_path(source.location, roots)
    name = _safe_filename(source.original_name if source.original_name is not None else path.name)
    try:
        size = path.stat().st_size
    except OSError:
        raise InputValidationError("FILE_NOT_FOUND", "Input file became unavailable") from None
    if size < 1:
        raise InputValidationError("EMPTY_FILE", "Audio file is empty")
    if size > current_limits.max_file_bytes:
        raise InputValidationError("FILE_TOO_LARGE", "Audio file exceeds the configured size limit")
    duration, codec = _probe_audio(path, binary=ffprobe_binary,
                                   timeout=current_limits.ffprobe_timeout_seconds)
    if duration > current_limits.max_duration_seconds:
        raise InputValidationError("AUDIO_TOO_LONG", "Audio duration exceeds the configured limit")
    return ValidatedInput(
        source_kind=source.kind, path=path, safe_filename=name,
        storage_filename=f"{uuid4().hex}-{name}", size_bytes=size,
        duration_seconds=duration, codec_name=codec,
    )

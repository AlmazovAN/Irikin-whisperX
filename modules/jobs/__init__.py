"""Pure-Python input, durable job and headless transcription contracts."""

from .inputs import (
    InputKind, InputLimits, InputSource, InputValidationError, ValidatedInput,
    validate_input,
)

from .transcription_service import (
    TranscriptionResult, TranscriptionService, TranscriptionServiceError,
)

from .service import JobIntakeError, JobService, JobTicket
from .storage import DEFAULT_JOB_ROOT, JobStorage, JobStorageError, StoredAudio

from .results import (
    ResultFiles, ResultFormatError, SCHEMA_VERSION, build_result_payload,
    normalize_segments, render_json, render_txt, write_result_files,
)

__all__ = (
    "InputKind", "InputLimits", "InputSource", "InputValidationError",
    "ValidatedInput", "validate_input",
    "JobIntakeError", "JobService", "JobTicket",
    "DEFAULT_JOB_ROOT", "JobStorage", "JobStorageError", "StoredAudio",
    "TranscriptionResult", "TranscriptionService", "TranscriptionServiceError",
    "ResultFiles", "ResultFormatError", "SCHEMA_VERSION", "build_result_payload",
    "normalize_segments", "render_json", "render_txt", "write_result_files",
)

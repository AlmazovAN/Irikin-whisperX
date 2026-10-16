"""Pure-Python job intake contracts (worker and persistence added later)."""

from .inputs import (
    InputKind, InputLimits, InputSource, InputValidationError, ValidatedInput,
    validate_input,
)

from .transcription_service import (
    TranscriptionResult, TranscriptionService, TranscriptionServiceError,
)

from .results import (
    ResultFiles, ResultFormatError, SCHEMA_VERSION, build_result_payload,
    normalize_segments, render_json, render_txt, write_result_files,
)

__all__ = (
    "InputKind", "InputLimits", "InputSource", "InputValidationError",
    "ValidatedInput", "validate_input",
    "TranscriptionResult", "TranscriptionService", "TranscriptionServiceError",
    "ResultFiles", "ResultFormatError", "SCHEMA_VERSION", "build_result_payload",
    "normalize_segments", "render_json", "render_txt", "write_result_files",
)

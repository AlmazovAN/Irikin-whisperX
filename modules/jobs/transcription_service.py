"""P09: headless adapter for the existing WhisperX transcription pipeline.

The service only accepts a P08 ValidatedInput. It does not perform an upload,
write results, start a background thread or create a second ASR implementation.
P15 will own a single instance of this service inside the durable worker.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Callable

from modules.jobs.inputs import ValidatedInput
from modules.mvp.preset import MVP_PRESET, build_mvp_pipeline_params

if TYPE_CHECKING:
    from modules.whisper.data_classes import Segment

logger = logging.getLogger(__name__)


class TranscriptionServiceError(RuntimeError):
    """Stable, non-sensitive diagnostic code for a failed transcription."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class HeadlessProgress:
    """The legacy pipeline expects a callable Gradio progress object.

    Only its callable interface is used by the reused WhisperX pipeline. It
    deliberately emits no browser events and makes no ETA claims.
    """

    def __call__(self, value: float | int, *, desc: str | None = None) -> None:
        pass


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Existing Segment objects, unchanged; normalized JSON belongs to P12."""

    segments: tuple[Segment, ...]
    elapsed_seconds: float
    input_name: str
    model_size: str
    language: str
    diarization_requested: bool

    @property
    def diagnostic_only(self) -> bool:
        return not self.diarization_requested


def _new_whisperx_pipeline():
    """Lazy import: importing this service must not load ML libraries."""
    from modules.whisper.whisper_factory import WhisperFactory

    return WhisperFactory.create_whisper_inference(whisper_type="whisperx")


def _verify_source(source: ValidatedInput) -> Path:
    if not isinstance(source, ValidatedInput):
        raise TypeError("TranscriptionService requires a validated P08 input")
    try:
        canonical = source.path.resolve(strict=True)
        if canonical != source.path or not canonical.is_file():
            raise OSError("Input changed")
        if canonical.stat().st_size != source.size_bytes:
            raise OSError("Input size changed")
    except (OSError, ValueError, RuntimeError):
        raise TranscriptionServiceError(
            "INPUT_CHANGED", "Validated audio is missing or changed; validate again"
        ) from None
    return canonical


def _verify_segments(segments: object, *, require_speakers: bool) -> tuple[Segment, ...]:
    if not isinstance(segments, (list, tuple)) or not segments:
        raise TranscriptionServiceError("EMPTY_TRANSCRIPT", "WhisperX returned no transcription")
    clean = tuple(segments)
    spoken = [item for item in clean if isinstance(getattr(item, "text", None), str)
              and item.text.strip()]
    if not spoken:
        raise TranscriptionServiceError("EMPTY_TRANSCRIPT", "WhisperX returned no recognized speech")

    def valid_times(item: object) -> bool:
        start, end = getattr(item, "start", None), getattr(item, "end", None)
        try:
            return (start is not None and end is not None and
                    math.isfinite(float(start)) and math.isfinite(float(end)) and
                    0 <= float(start) <= float(end))
        except (ValueError, TypeError, OverflowError):
            return False

    if any(not valid_times(s) for s in spoken):
        raise TranscriptionServiceError("INVALID_SEGMENTS", "Recognized segments lack valid timestamps")
    words = [word for segment in spoken for word in (getattr(segment, "words", None) or ())]
    if not any(valid_times(w) for w in words):
        raise TranscriptionServiceError("ALIGNMENT_MISSING", "WhisperX alignment returned no word timestamps")
    if require_speakers and not any(
        (isinstance(getattr(s, "speaker", None), str) and s.speaker.strip())
        for s in spoken
    ) and not any(
        (isinstance(getattr(w, "speaker", None), str) and w.speaker.strip())
        for w in words
    ):
        raise TranscriptionServiceError("DIARIZATION_MISSING", "Speaker labels were not assigned")
    return clean


class TranscriptionService:
    """A reusable, non-queuing owner of one existing WhisperX inference instance.

    P15 will serialize jobs through its worker. Until then, a nonblocking lock
    refuses a concurrent call rather than silently running two heavy models.
    """

    def __init__(self, *, pipeline_factory: Callable[[], object] | None = None) -> None:
        self._factory = pipeline_factory or _new_whisperx_pipeline
        self._pipeline: object | None = None
        self._lock = Lock()

    def transcribe(
        self,
        source: ValidatedInput,
        *,
        device: str = MVP_PRESET.default_device,
        compute_type: str = MVP_PRESET.default_compute_type,
        batch_size: int = MVP_PRESET.default_batch_size,
        diarization_enabled: bool = MVP_PRESET.diarization_enabled,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """Call BaseTranscriptionPipeline.run without a Gradio request/session.

        All model settings come from P07; no configuration YAML is persisted.
        Diarization disabled is a diagnostic-only, NOT accepted MVP, result.
        """
        source_path = _verify_source(source)
        params = build_mvp_pipeline_params(
            device=device, compute_type=compute_type,
            batch_size=batch_size, diarization_enabled=diarization_enabled,
        )
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable")
        if not self._lock.acquire(blocking=False):
            raise TranscriptionServiceError("BUSY", "Another transcription is already running")
        pipeline = None
        try:
            try:
                if self._pipeline is None:
                    self._pipeline = self._factory()
                pipeline = self._pipeline
                pipeline.device = device
                raw, elapsed = pipeline.run(
                    str(source_path), HeadlessProgress(), MVP_PRESET.output_format,
                    False, progress_callback, *params.to_list(), persist_settings=False,
                )
            except Exception:
                raise TranscriptionServiceError(
                    "PIPELINE_FAILED", "WhisperX inference failed; inspect local diagnostics"
                ) from None
            segments = _verify_segments(raw, require_speakers=diarization_enabled)
            try:
                elapsed_number = float(elapsed)
            except (TypeError, ValueError, OverflowError):
                elapsed_number = float("nan")
            if not math.isfinite(elapsed_number) or elapsed_number < 0:
                raise TranscriptionServiceError("INVALID_DURATION", "WhisperX returned invalid elapsed time")
            return TranscriptionResult(
                segments=segments, elapsed_seconds=elapsed_number,
                input_name=source.safe_filename,
                model_size=MVP_PRESET.model_size, language=MVP_PRESET.language,
                diarization_requested=diarization_enabled,
            )
        finally:
            try:
                if pipeline is not None:
                    pipeline.offload()
                    wrapper = getattr(pipeline, "whisperx_wrapper", None)
                    if wrapper is not None:
                        wrapper.offload_diarizer()
            except Exception:
                # No traceback/credentials in logs. Discard potentially dirty owner.
                self._pipeline = None
                logger.warning("WhisperX cleanup failed; the inference instance will be replaced")
            finally:
                self._lock.release()

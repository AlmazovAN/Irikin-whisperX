"""P09: headless adapter for the existing WhisperX transcription pipeline.

The service only accepts a P08 ValidatedInput. It does not perform an upload,
write results, start a background thread or create a second ASR implementation.
P15 will own a single instance of this service inside the durable worker.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import logging
import math
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Callable

from modules.jobs.inputs import ValidatedInput
from modules.diarize.diagnostics import ModelStageError, require_hf_token
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
    """Consume trusted stage events from existing Gradio-style progress calls.

    This emits only known canonical stage identifiers, never arbitrary text,
    browser updates or ETAs. Duplicate consecutive events are coalesced.
    """

    def __init__(self, stage_callback: Callable[[str], None] | None = None):
        self._stage_callback = stage_callback
        self._last_stage: str | None = None

    def __call__(self, value: float | int, *, desc: str | None = None) -> None:
        from modules.jobs.progress import stage_from_description

        stage = stage_from_description(desc)
        if stage is not None and stage != self._last_stage:
            self._last_stage = stage
            if self._stage_callback is not None:
                self._stage_callback(stage)


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Existing Segment objects, unchanged; PR13 offers explicit normalized export."""

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
        stage_callback: Callable[[str], None] | None = None,
        cancel_event: object | None = None,
    ) -> TranscriptionResult:
        """Call BaseTranscriptionPipeline.run without a Gradio request/session.

        All model settings come from P07; no configuration YAML is persisted.
        Diarization disabled is a diagnostic-only, NOT accepted MVP, result.
        """
        if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
            raise TypeError("cancel_event must provide is_set()")

        def check_cancel() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise TranscriptionServiceError("CANCELLED", "Transcription was cancelled")

        check_cancel()
        source_path = _verify_source(source)
        params = build_mvp_pipeline_params(
            device=device, compute_type=compute_type,
            batch_size=batch_size, diarization_enabled=diarization_enabled,
        )
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable")
        if stage_callback is not None and not callable(stage_callback):
            raise TypeError("stage_callback must be callable")
        check_cancel()
        if diarization_enabled:
            try:
                require_hf_token(os.environ.get("HF_TOKEN"))
            except ModelStageError as exc:
                raise TranscriptionServiceError(exc.code, str(exc)) from None
        if not self._lock.acquire(blocking=False):
            raise TranscriptionServiceError("BUSY", "Another transcription is already running")
        pipeline = None
        previous_event = None
        event_installed = False
        try:
            check_cancel()
            try:
                if self._pipeline is None:
                    self._pipeline = self._factory()
                pipeline = self._pipeline
                pipeline.device = device
                setter = getattr(pipeline, "set_cancel_event", None)
                if cancel_event is not None and callable(setter):
                    previous_event = getattr(pipeline, "cancel_event", None)
                    setter(cancel_event)
                    event_installed = True

                def checked_stage(stage: str) -> None:
                    check_cancel()
                    if stage_callback is not None:
                        stage_callback(stage)

                def checked_progress(fraction: float) -> None:
                    check_cancel()
                    if progress_callback is not None:
                        progress_callback(fraction)

                raw, elapsed = pipeline.run(
                    str(source_path), HeadlessProgress(checked_stage), MVP_PRESET.output_format,
                    False, checked_progress, *params.to_list(),
                    persist_settings=False, strict_vad=True,
                )
                check_cancel()
            except ModelStageError as exc:
                self._pipeline = None  # Do not reuse a potentially dirty instance.
                check_cancel()
                # Keep only known, safe pipeline diagnostic codes.
                raise TranscriptionServiceError(exc.code, str(exc)) from None
            except Exception as exc:
                self._pipeline = None  # Always offload in finally before reuse.
                check_cancel()
                if isinstance(exc, TranscriptionServiceError):
                    raise
                # Existing WhisperXWrapper handles recoverable CUDA OOM itself.
                # An exhausted fallback is not a successful transcription.
                description = str(exc).lower()
                code = "OUT_OF_MEMORY" if (
                    isinstance(exc, MemoryError)
                    or "out of memory" in description
                    or "cudnn_status_alloc_failed" in description
                    or "defaultcpuallocator" in description
                ) else "PIPELINE_FAILED"
                raise TranscriptionServiceError(
                    code, "WhisperX inference failed; review local diagnostics"
                ) from None
            check_cancel()
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
                if event_installed and pipeline is not None:
                    pipeline.set_cancel_event(previous_event)
            except Exception:
                self._pipeline = None
                logger.warning("WhisperX cancel signal cleanup failed; discarding owner")
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

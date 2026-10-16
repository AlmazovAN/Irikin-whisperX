"""PR17 / P16: honest milestone-based progress for the existing WhisperX pipeline.

Stored Task.progress is a monotonically increasing UI indicator, NOT elapsed
audio fraction, measured inference completion, or remaining-time estimate.
"""
from __future__ import annotations

import math

STAGES = ("PREPARING", "VAD", "ASR", "ALIGN", "DIARIZE", "EXPORT")
STAGE_START: dict[str, float] = {
    "PREPARING": 0.0,
    "VAD": 0.03,
    "ASR": 0.12,
    "ALIGN": 0.72,
    "DIARIZE": 0.85,
    "EXPORT": 0.95,
}
_STAGE_BY_MESSAGE = {
    "Filtering silent parts from audio..": "VAD",
    "Transcribing with WhisperX..": "ASR",
    "Aligning with WhisperX..": "ALIGN",
    "Diarizing speakers..": "DIARIZE",
}


def stage_from_description(description: str | None) -> str | None:
    """Only recognize exact, known stage messages; never store arbitrary text."""
    return _STAGE_BY_MESSAGE.get(description) if isinstance(description, str) else None


def progress_for_stage(stage: str, *, asr_fraction: float | None = None) -> float:
    """Map actual stage events to monotonic UI milestones (<1 until success).

    WhisperX's ASR callback reports an approximate fraction of processed audio,
    not of whole-pipeline work. The ASR progress segment is [0.12, 0.70].
    """
    if stage not in STAGE_START:
        raise ValueError("Unknown job processing stage")
    floor = STAGE_START[stage]
    if stage != "ASR" or asr_fraction is None:
        return floor
    if isinstance(asr_fraction, bool) or not isinstance(asr_fraction, (int, float)):
        raise ValueError("Invalid ASR progress value")
    fraction = float(asr_fraction)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("ASR progress must be a finite fraction from 0 to 1")
    return round(floor + fraction * (0.70 - floor), 6)


def stage_index(stage: str) -> int:
    if stage not in STAGE_START:
        raise ValueError("Unknown job processing stage")
    return STAGES.index(stage)

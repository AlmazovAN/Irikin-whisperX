"""Restore source-audio timestamps after Silero VAD concatenates speech chunks.

Chunk coordinates are sample offsets in ORIGINAL 16 kHz audio. WhisperX ASR
runs on their concatenation; alignment and diarization use the original audio.
This stdlib-only module is unit-testable without torch, Gradio or model weights.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True, slots=True)
class _Chunk:
    start: int
    end: int
    trimmed_start: int
    trimmed_end: int


class SpeechTimeMap:
    """Piecewise monotonic map from concatenated VAD audio to original time.

    A seam is mapped to the NEXT chunk for the start of a span, and to the
    PREVIOUS chunk for its end; pauses are never erased from output timestamps.
    """

    def __init__(self, speech_chunks: Sequence[dict], sample_rate: int = 16000):
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if not speech_chunks:
            raise ValueError("speech_chunks must not be empty")
        self.sample_rate = sample_rate
        chunks: list[_Chunk] = []
        total = 0
        previous_end = 0
        for raw in speech_chunks:
            if not isinstance(raw, dict):
                raise ValueError("VAD chunk must be a dict")
            begin, end = raw.get("start"), raw.get("end")
            if (type(begin) is not int or type(end) is not int
                    or begin < previous_end or end <= begin):
                raise ValueError("VAD chunks must be ordered, non-overlapping sample ranges")
            length = end - begin
            chunks.append(_Chunk(begin, end, total, total + length))
            total += length
            previous_end = end
        self._chunks = tuple(chunks)
        self._trimmed_ends = tuple(c.trimmed_end for c in chunks)
        self._total_samples = total

    def original_time(self, seconds: float, *, endpoint: str = "start") -> float:
        if endpoint not in ("start", "end"):
            raise ValueError("endpoint must be 'start' or 'end'")
        try:
            time = float(seconds)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("Timestamp is not numeric") from exc
        if not math.isfinite(time) or time < 0:
            raise ValueError("Timestamp must be finite and nonnegative")
        trimmed_position = time * self.sample_rate
        # ASR can round the trimmed end by a fraction of a sample.
        tolerance = 1.0
        if trimmed_position > self._total_samples + tolerance:
            raise ValueError("Timestamp lies outside VAD audio duration")
        trimmed_position = min(trimmed_position, self._total_samples)
        if endpoint == "start":
            index = bisect_right(self._trimmed_ends, trimmed_position)
        else:
            index = bisect_left(self._trimmed_ends, trimmed_position)
        chunk = self._chunks[min(index, len(self._chunks) - 1)]
        relative = max(0.0, min(trimmed_position - chunk.trimmed_start,
                                chunk.trimmed_end - chunk.trimmed_start))
        return (chunk.start + relative) / self.sample_rate


def restore_original_timestamps(
    segments: Sequence[object],
    speech_chunks: Sequence[dict],
    sample_rate: int = 16000,
) -> list[object]:
    """Restore segment and optional word timestamps in-place.

    Missing word timestamps remain missing. Segment bounds are restored even
    when a word has no timestamps. A word straddling a VAD seam spans the
    original pause; each endpoint is mapped independently to avoid collapsing
    the gap. Word-aligned results are subsequently refined on original audio.
    """
    if not segments:
        return list(segments)
    time_map = SpeechTimeMap(speech_chunks, sample_rate)
    for segment in segments:
        start, end = getattr(segment, "start", None), getattr(segment, "end", None)
        if start is not None:
            segment.start = time_map.original_time(start, endpoint="start")
        if end is not None:
            segment.end = time_map.original_time(end, endpoint="end")
        valid_words = []
        for word in (getattr(segment, "words", None) or ()):
            word_start, word_end = getattr(word, "start", None), getattr(word, "end", None)
            if word_start is None or word_end is None:
                continue
            word.start = time_map.original_time(word_start, endpoint="start")
            word.end = time_map.original_time(word_end, endpoint="end")
            valid_words.append(word)
        if valid_words:
            segment.start = valid_words[0].start
            segment.end = valid_words[-1].end
    return list(segments)

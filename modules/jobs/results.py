"""PR13: stable, dependency-free JSON/TXT representation of WhisperX Segment.

Only data *returned* by the existing ASR pipeline is normalized here. The
legacy Gradio subtitle writer still receives original Segment objects. Output
files are opt-in and NOT a replacement for persistent job storage (PR14+).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4


SCHEMA_VERSION = 1
_LEGACY_SPEAKER = re.compile(r"^(SPEAKER_[0-9]+)\s*\|\s*")
_SAFE_STEM = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


class ResultFormatError(ValueError):
    """Invalid segment or metadata, with no user data in its message."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ResultFiles:
    json_path: Path
    txt_path: Path


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _time(value: object, *, required: bool = True) -> float | None:
    if value is None:
        if not required:
            return None
        raise ResultFormatError("INVALID_TIMESTAMP", "Segment timestamp is missing")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultFormatError("INVALID_TIMESTAMP", "Timestamp must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ResultFormatError("INVALID_TIMESTAMP", "Timestamp must be finite and nonnegative")
    return value


def _speaker(value: object) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ResultFormatError("INVALID_SPEAKER", "Speaker label must be text")
    return value.strip() or None


def _word(raw: object) -> dict[str, Any] | None:
    text = _field(raw, "word", _field(raw, "text"))
    if not isinstance(text, str):
        raise ResultFormatError("INVALID_WORD", "Word text must be a string")
    # WhisperX may prefix words with spaces. Keep the significant internal text.
    text = text.strip()
    if not text:
        return None
    data: dict[str, Any] = {"word": text}
    start = _time(_field(raw, "start"), required=False)
    end = _time(_field(raw, "end"), required=False)
    if start is not None and end is not None and start > end:
        raise ResultFormatError("INVALID_TIMESTAMP", "Word start is later than end")
    if start is not None:
        data["start"] = start
    if end is not None:
        data["end"] = end
    speaker = _speaker(_field(raw, "speaker"))
    if speaker is not None:
        data["speaker"] = speaker

    # Use the first *present* score. 0.0 is valid and must never be lost via 'or'.
    for key in ("confidence", "probability", "score"):
        confidence = _field(raw, key)
        if confidence is not None:
            if (isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or not math.isfinite(float(confidence))
                    or not 0.0 <= float(confidence) <= 1.0):
                raise ResultFormatError("INVALID_CONFIDENCE", "Word confidence must be between 0 and 1")
            data["confidence"] = float(confidence)
            break
    return data


def normalize_segments(segments: Sequence[object]) -> list[dict[str, Any]]:
    """Return a fresh serializable list. Never modify original Segment or Word.

    Legacy "SPEAKER_00|..." prefixes are removed only if matching the separate
    speaker field, or if there is no speaker field to contradict that prefix.
    The old subtitle writer remains responsible for its original display.
    """
    if isinstance(segments, (str, bytes)) or not isinstance(segments, (list, tuple)):
        raise ResultFormatError("INVALID_SEGMENTS", "Expected a sequence of segments")
    normalized: list[dict[str, Any]] = []
    for raw in segments:
        text = _field(raw, "text")
        if text is None or text == "":
            continue  # Empty legacy placeholder for silence; not a spoken segment.
        if not isinstance(text, str):
            raise ResultFormatError("INVALID_TEXT", "Segment text must be a string")
        text = text.strip()
        if not text:
            continue
        start, end = _time(_field(raw, "start")), _time(_field(raw, "end"))
        if start > end:
            raise ResultFormatError("INVALID_TIMESTAMP", "Segment start is later than end")
        speaker = _speaker(_field(raw, "speaker"))
        prefix = _LEGACY_SPEAKER.match(text)
        if prefix is not None and (speaker is None or speaker == prefix.group(1)):
            speaker = speaker or prefix.group(1)
            text = text[prefix.end():].strip()
        elif speaker is None and text.startswith("None|"):
            # Legacy Diarizer prepends "None|" when there was no speaker match.
            text = text[5:].strip()
        if not text:
            continue
        raw_words = _field(raw, "words") or ()
        if isinstance(raw_words, (str, bytes)) or not isinstance(raw_words, (list, tuple)):
            raise ResultFormatError("INVALID_WORD", "Segment words must be a sequence")
        words = [item for raw_word in raw_words if (item := _word(raw_word)) is not None]
        if speaker is None and words:
            distinct = {item["speaker"] for item in words if item.get("speaker")}
            if len(distinct) == 1:
                speaker = next(iter(distinct))
        item: dict[str, Any] = {
            "start": start,
            "end": end,
            "text": text,
            "speaker": speaker,
        }
        if words:
            item["words"] = words
        normalized.append(item)
    return normalized


def build_result_payload(result: object) -> dict[str, Any]:
    """Convert a successful PR09/PR12 TranscriptionResult into schema v1."""
    segments = normalize_segments(_field(result, "segments"))
    if not segments:
        raise ResultFormatError("EMPTY_TRANSCRIPT", "No spoken segments to export")
    elapsed = _time(_field(result, "elapsed_seconds"))
    model = _field(result, "model_size")
    language = _field(result, "language")
    if not isinstance(model, str) or not model.strip():
        raise ResultFormatError("INVALID_METADATA", "Model is missing")
    if not isinstance(language, str) or not language.strip():
        raise ResultFormatError("INVALID_METADATA", "Language is missing")
    name = _field(result, "input_name")
    if not isinstance(name, str) or not name:
        raise ResultFormatError("INVALID_METADATA", "Input name is missing")
    # No system paths, environment values or raw HF tokens in the output JSON.
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    if name in ("", ".", ".."):
        raise ResultFormatError("INVALID_METADATA", "Input name is invalid")
    requested = _field(result, "diarization_requested")
    if not isinstance(requested, bool):
        raise ResultFormatError("INVALID_METADATA", "Diarization metadata must be boolean")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_name": name,
        "model_size": model,
        "language": language,
        "elapsed_seconds": elapsed,
        "diarization_requested": requested,
        "diagnostic_only": not requested,
        "segments": segments,
    }


def render_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def _timestamp(seconds: float) -> str:
    """Floor segment START to whole seconds; never move an utterance forward.

    Precise start/end (including milliseconds) remain in transcript.json.
    """
    total_seconds = int(math.floor(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def render_txt(payload: Mapping[str, Any]) -> str:
    """MVP transcript: [HH:MM:SS] [SPEAKER_00]: speech.

    The separate speaker field is authoritative; unknown speakers receive no
    invented identifier. This writer is independent of legacy subtitle TXT.
    """
    lines: list[str] = []
    for item in payload["segments"]:
        start = _time(item.get("start"))
        speaker = _speaker(item.get("speaker"))
        tag = f"[{speaker}]: " if speaker is not None else ""
        text = " ".join(item["text"].split())  # Keep each segment on one line.
        lines.append(f"[{_timestamp(start)}] {tag}{text}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_result_files(
    result: object, output_dir: str | Path, *, stem: str | None = None
) -> ResultFiles:
    """Explicit one-shot export; no job lifecycle or disk directory policy.

    Caller supplies a trusted existing directory. "x" prevents accidental
    overwrites; both newly-created files are removed if either write fails.
    PR14+ will replace this with durable per-job atomic storage.
    """
    payload = build_result_payload(result)
    output = Path(output_dir)
    if not output.is_dir():
        raise ResultFormatError("INVALID_OUTPUT_DIR", "Export directory must exist")
    prefix = stem if stem is not None else "result-" + uuid4().hex
    if not isinstance(prefix, str) or not _SAFE_STEM.fullmatch(prefix):
        raise ResultFormatError("INVALID_FILENAME", "Export stem must be a safe basename")
    json_path, txt_path = output / (prefix + ".json"), output / (prefix + ".txt")
    created: list[Path] = []
    try:
        with json_path.open("x", encoding="utf-8", newline="\n") as stream:
            created.append(json_path)
            stream.write(render_json(payload))
        with txt_path.open("x", encoding="utf-8", newline="\n") as stream:
            created.append(txt_path)
            stream.write(render_txt(payload))
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return ResultFiles(json_path=json_path, txt_path=txt_path)

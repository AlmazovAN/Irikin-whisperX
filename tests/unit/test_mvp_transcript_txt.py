"""P22: MVP transcript TXT contract and durable worker/artifact integration.

Offline: real SQLite/files/thread; fake FFprobe/WhisperX; no audio leaves disk.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import time
import wave

import pytest

from modules.jobs.artifacts import publish_result_set, verify_result_set
from modules.jobs.inputs import InputSource
from modules.jobs.results import (
    ResultFormatError, build_result_payload, render_txt, write_result_files,
)
from modules.jobs.service import JobService
from modules.jobs.worker import JobWorker

pytestmark = pytest.mark.fast
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Segment:
    start: float | None
    end: float | None
    text: str
    speaker: str | None = None
    words: tuple = ()


@dataclass
class Result:
    segments: tuple
    input_name: str = "совещание.wav"
    elapsed_seconds: float = 1.0
    model_size: str = "small"
    language: str = "ru"
    diarization_requested: bool = True


@pytest.fixture
def meeting():
    return Result(segments=(
        Segment(12.345, 14.8, "SPEAKER_00|Добрый день", "SPEAKER_00"),
        Segment(60.999, 63.2, "SPEAKER_01|Здравствуйте", "SPEAKER_01"),
        Segment(3601.6, 3604.25, "Продолжим\n  обсуждение", None),
    ))


def test_txt_uses_start_time_and_bracketed_speaker_without_inventing_unknown(meeting):
    payload = build_result_payload(meeting)
    assert render_txt(payload) == (
        "[00:00:12] [SPEAKER_00]: Добрый день\n"
        "[00:01:00] [SPEAKER_01]: Здравствуйте\n"
        "[01:00:01] Продолжим обсуждение\n"
    )
    # Full precision and explicit unknown speaker are preserved in JSON.
    assert payload["segments"][0]["start"] == 12.345
    assert payload["segments"][0]["end"] == 14.8
    assert payload["segments"][2]["speaker"] is None
    assert payload["segments"][2]["text"] == "Продолжим\n  обсуждение"
    assert "SPEAKER_00|" not in render_txt(payload)


@pytest.mark.parametrize(("seconds", "expected"), [
    (0.0, "00:00:00"), (0.999, "00:00:00"),
    (59.999, "00:00:59"), (60.0, "00:01:00"),
    (3599.999, "00:59:59"), (3600.0, "01:00:00"),
    (25 * 3600 + 1.0, "25:00:01"),
])
def test_timestamp_floor_never_moves_speech_after_its_real_start(seconds, expected):
    result = Result(segments=(Segment(seconds, seconds + 1, "Текст", "SPEAKER_00"),))
    assert render_txt(build_result_payload(result)) == (
        f"[{expected}] [SPEAKER_00]: Текст\n"
    )


def test_missing_start_does_not_make_a_fake_timestamp_or_publish_result(tmp_path):
    result = Result(segments=(Segment(None, 3, "Слова", None),))
    with pytest.raises(ResultFormatError) as error:
        build_result_payload(result)
    assert error.value.code == "INVALID_TIMESTAMP"
    job_folder = tmp_path / "job"
    job_folder.mkdir()
    with pytest.raises(ResultFormatError) as error:
        publish_result_set(result, job_folder, attempt=1)
    assert error.value.code == "INVALID_TIMESTAMP"
    assert not (job_folder / "result").exists()


def test_one_shot_export_and_durable_bundle_share_exact_formatter(meeting, tmp_path):
    standalone = tmp_path / "one_shot"
    standalone.mkdir()
    files = write_result_files(meeting, standalone, stem="meeting")
    job_folder = tmp_path / "job"
    job_folder.mkdir()
    published = publish_result_set(meeting, job_folder, attempt=1)
    assert verify_result_set(published.manifest_path.parent, attempt=1) == published
    assert files.json_path.read_bytes() == published.files.json_path.read_bytes()
    assert files.txt_path.read_bytes() == published.files.txt_path.read_bytes()
    payload = json.loads(published.files.json_path.read_text(encoding="utf-8"))
    assert published.files.txt_path.read_text(encoding="utf-8") == render_txt(payload)


def test_legacy_plain_subtitle_writer_remains_independent():
    path = ROOT / "modules" / "utils" / "subtitle_manager.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WriteTXT")
    method = next(n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == "write_result")
    source = ast.get_source_segment(path.read_text(encoding="utf-8"), method)
    assert 'segment["text"].strip()' in source
    assert "render_txt" not in source  # Legacy users still get their plain subtitles.


def test_completed_worker_result_is_readable_from_jobservice_with_same_txt(
    meeting, tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        "modules.jobs.inputs._probe_audio",
        lambda path, *, binary, timeout: (4.0, "pcm_s16le"),
    )
    approved = tmp_path / "approved"
    approved.mkdir()
    audio = approved / "meeting.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 8000)

    service = JobService(
        storage_root=tmp_path / "jobs", local_roots=(approved,),
    )
    ticket = service.submit(InputSource.local_file(audio))

    class FakeTranscriber:
        def transcribe(self, source):
            assert source.safe_filename == "meeting.wav"
            return meeting

    worker = JobWorker(
        service, transcriber_factory=FakeTranscriber, poll_interval=0.01,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 6
        state = service.get_status(ticket.identifier)
        while state.status not in ("COMPLETED", "FAILED") and time.monotonic() < deadline:
            time.sleep(0.01)
            state = service.get_status(ticket.identifier)
        assert state.status == "COMPLETED", state.error_code
    finally:
        assert worker.stop(timeout=5)

    txt = service.get_artifact(ticket.identifier, format="txt").read_text("utf-8")
    js = json.loads(service.get_artifact(ticket.identifier, format="json").read_text("utf-8"))
    assert txt == render_txt(js)
    assert js["segments"][1]["speaker"] == "SPEAKER_01"
    assert txt.startswith("[00:00:12] [SPEAKER_00]: Добрый день\n")

"""PR17 / P16: durable status and real-stage events (no Torch/model downloads)."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Event
import time
import wave

import pytest

from modules.jobs.inputs import InputSource
from modules.jobs.progress import STAGE_START, progress_for_stage, stage_from_description
from modules.jobs.service import JobService
from modules.jobs.transcription_service import HeadlessProgress
from modules.jobs.worker import JobWorker, _Claimed

pytestmark = pytest.mark.fast


@pytest.fixture(autouse=True)
def fake_ffprobe(monkeypatch):
    monkeypatch.setattr(
        "modules.jobs.inputs._probe_audio",
        lambda path, *, binary, timeout: (0.5, "pcm_s16le"),
    )


@pytest.fixture
def jobs(tmp_path):
    allowed = tmp_path / "approved"
    allowed.mkdir()
    root = tmp_path / "data" / "jobs"
    service = JobService(storage_root=root, local_roots=(allowed,))
    def create(name="meeting"):
        audio = allowed / f"{name}.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\0\0" * 8000)
        return service.submit(InputSource.local_file(audio))
    return service, root, create


@dataclass
class Segment:
    start: float = 0.0
    end: float = 0.45
    text: str = "Привет"
    speaker: str = "SPEAKER_00"
    words: tuple = ()


@dataclass
class Result:
    input_name: str
    segments: tuple = (Segment(),)
    elapsed_seconds: float = 0.5
    model_size: str = "small"
    language: str = "ru"
    diarization_requested: bool = True


def wait_for(service, identifier, stage=None, status=None):
    until = time.monotonic() + 6
    while time.monotonic() < until:
        value = service.get_status(identifier)
        if value and (stage is None or value.stage == stage) and (
            status is None or value.status == status
        ):
            return value
        time.sleep(0.01)
    pytest.fail(f"Status did not reach {stage or status}")


def test_pipeline_headless_progress_maps_only_real_stage_messages():
    seen = []
    progress = HeadlessProgress(stage_callback=seen.append)
    progress(0.0, desc="Filtering silent parts from audio..")
    progress(0.2, desc="Filtering silent parts from audio..")
    progress(0.0, desc="Transcribing with WhisperX..")
    progress(0.3, desc="Transcribing with WhisperX..")
    progress(0.95, desc="Aligning with WhisperX..")
    progress(0.99, desc="Diarizing speakers..")
    progress(1, desc="Finished.")
    progress(0.4, desc="secret error /tmp/sensitive")
    assert seen == ["VAD", "ASR", "ALIGN", "DIARIZE"]
    assert stage_from_description("Retrying WhisperX on CPU after CUDA OOM..") is None


@pytest.mark.parametrize("value", [-1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_model_progress_is_rejected(value):
    with pytest.raises(ValueError):
        progress_for_stage("ASR", asr_fraction=value)


def test_progress_scale_is_monotone_but_never_estimates_eta():
    marks = [progress_for_stage(s) for s in STAGE_START]
    assert marks == sorted(marks)
    assert marks[0] == 0
    assert marks[-1] < 1
    assert progress_for_stage("ASR", asr_fraction=0.5) == 0.41
    assert progress_for_stage("ASR", asr_fraction=1.0) < STAGE_START["ALIGN"]


def test_get_status_is_fresh_on_new_service_instance_and_is_path_free(jobs):
    svc, root, add = jobs
    item = add()
    reopened = JobService(storage_root=root)
    assert svc.get_status(item.identifier) == reopened.get_status(item.identifier)
    assert reopened.get_status(item.identifier) == item
    assert item.progress == 0
    assert item.updated_at == item.created_at
    assert item.started_at is None and item.finished_at is None
    assert not hasattr(item, "input_path")
    assert not hasattr(item, "output_json_path")
    assert reopened.get_status("invalid-id") is None


def test_real_sqlite_updates_are_monotone_and_bound_to_attempt(jobs):
    svc, _, add = jobs
    item = add()
    w = JobWorker(svc)
    claim = w._claim()
    assert claim and claim.identifier == item.identifier
    assert svc.get_status(item.identifier).stage == "PREPARING"
    assert svc.get_status(item.identifier).attempt == 1
    assert w._record_stage(claim, "VAD")
    assert w._record_stage(claim, "ASR", asr_fraction=0.5)
    assert not w._record_stage(claim, "VAD")
    assert not w._record_stage(claim, "ASR", asr_fraction=0.1)
    live = svc.get_status(item.identifier)
    assert live.stage == "ASR" and live.progress == 0.41
    assert w._record_stage(claim, "ALIGN")
    assert w._record_stage(claim, "DIARIZE")
    assert not w._record_stage(claim, "ASR", asr_fraction=1.0)
    assert w._record_stage(claim, "EXPORT")
    assert svc.get_status(item.identifier).progress == 0.95
    assert not w._record_stage(_Claimed(item.identifier, 9), "EXPORT")
    w._finish(claim, success=True)
    completed = svc.get_status(item.identifier)
    assert completed.status == "COMPLETED"
    assert completed.progress == 1.0
    assert completed.stage == "DONE" and completed.finished_at
    assert not w._record_stage(claim, "ASR", asr_fraction=0.5)


def test_worker_stage_callbacks_visible_while_fake_pipeline_is_blocked(jobs):
    svc, root, add = jobs
    ticket = add("two_speakers")
    reached_asr, proceed = Event(), Event()
    class FakePipeline:
        def transcribe(self, source, *, stage_callback=None, progress_callback=None):
            stage_callback("VAD")
            stage_callback("ASR")
            progress_callback(0.25)
            reached_asr.set()
            if not proceed.wait(5):
                raise RuntimeError("test release timeout")
            progress_callback(0.8)
            stage_callback("ALIGN")
            stage_callback("DIARIZE")
            return Result(input_name=source.safe_filename)
    worker = JobWorker(svc, transcriber_factory=FakePipeline, poll_interval=0.01)
    worker.start()
    try:
        assert reached_asr.wait(4)
        # Another service simulates a new browser request/session polling DB.
        external = JobService(storage_root=root)
        running = external.get_status(ticket.identifier)
        assert running.status == "IN_PROGRESS"
        assert running.stage == "ASR"
        assert running.attempt == 1
        assert running.progress == 0.265
        assert running.started_at and running.updated_at
        assert running.progress < 1
        assert external.list_jobs()[0].progress == running.progress
        assert not hasattr(running, "input_path")
    finally:
        proceed.set()
        assert worker.stop(timeout=5)
    done = svc.get_status(ticket.identifier)
    assert done.status == "COMPLETED" and done.stage == "DONE"
    assert done.progress == 1.0
    assert done.finished_at is not None
    assert done.updated_at >= running.updated_at


def test_failed_job_retains_reached_milestone_until_explicit_retry(jobs):
    svc, _, add = jobs
    ticket = add("fail_at_diarization")
    class Failed:
        def transcribe(self, source, *, stage_callback=None):
            stage_callback("VAD")
            stage_callback("ASR")
            stage_callback("ALIGN")
            stage_callback("DIARIZE")
            raise RuntimeError("not safe to expose")
    worker = JobWorker(svc, transcriber_factory=Failed, poll_interval=0.01)
    worker.start()
    try:
        failed = wait_for(svc, ticket.identifier, status="FAILED")
    finally:
        assert worker.stop(timeout=5)
    assert failed.stage == "FAILED"
    assert failed.progress == STAGE_START["DIARIZE"]
    assert failed.error_code == "PROCESSING_FAILED"
    assert svc.retry(ticket.identifier)
    reset = svc.get_status(ticket.identifier)
    assert reset.status == "QUEUED"
    assert reset.stage == "QUEUED"
    assert reset.progress == 0
    assert reset.finished_at is None


def test_cancelled_queue_has_no_false_processing_progress(jobs):
    svc, _, add = jobs
    ticket = add()
    assert svc.cancel(ticket.identifier)
    status = svc.get_status(ticket.identifier)
    assert status.status == "CANCELLED"
    assert status.progress == 0.0
    assert status.finished_at is not None

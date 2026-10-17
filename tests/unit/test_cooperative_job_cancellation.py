"""PR20: durable cooperative cancellation and failure regression tests.

Real SQLite/threads/files; mocked ASR and FFprobe, without GPU or downloads.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from threading import Event
from pathlib import Path
import time
import wave

import pytest

from modules.jobs.artifacts import publish_result_set
from modules.jobs.inputs import InputSource, InputValidationError
from modules.jobs.service import JobIntakeError, JobService
from modules.jobs.worker import JobWorker

pytestmark = pytest.mark.fast


@dataclass
class Segment:
    start: float = 0.0
    end: float = 0.4
    text: str = "Проверка"
    speaker: str = "SPEAKER_00"
    words: tuple = ()


@dataclass
class Result:
    input_name: str = "private.wav"
    segments: tuple = (Segment(),)
    elapsed_seconds: float = 0.5
    model_size: str = "small"
    language: str = "ru"
    diarization_requested: bool = True


@pytest.fixture(autouse=True)
def ffprobe_stub(monkeypatch):
    monkeypatch.setattr(
        "modules.jobs.inputs._probe_audio",
        lambda path, *, binary, timeout: (0.4, "pcm_s16le"),
    )


@pytest.fixture
def env(tmp_path):
    uploaded = tmp_path / "allowed"
    uploaded.mkdir()
    root = tmp_path / "data" / "jobs"
    jobs = JobService(storage_root=root, local_roots=(uploaded,))

    def add(name):
        path = uploaded / (name + ".wav")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 6400)
        return jobs.submit(InputSource.local_file(path))

    return jobs, root, add


def await_state(jobs, identifier, status, *, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = jobs.get_status(identifier)
        if state and state.status == status:
            return state
        time.sleep(0.01)
    pytest.fail("Persisted job state did not reach expected status")


def begin(jobs, fake):
    worker = JobWorker(jobs, transcriber_factory=lambda: fake, poll_interval=0.01)
    worker.start()
    return worker


def test_cancel_queued_is_immediate_and_idempotent(env):
    jobs, _, add = env
    queued = add("queued")
    assert jobs.cancel(queued.identifier)
    assert not jobs.cancel(queued.identifier)
    done = jobs.get_status(queued.identifier)
    assert done.status == "CANCELLED"
    assert done.stage == "CANCELLED"
    assert done.error_code is None
    assert done.progress == 0
    assert done.finished_at
    assert not jobs.retry(queued.identifier)


def test_cancel_running_is_persisted_until_ASR_reaches_a_checkpoint(env):
    jobs, root, add = env
    slow = add("slow")
    next_job = add("next")
    entered, release = Event(), Event()
    seen = []

    class Blocking:
        def transcribe(self, source, *, cancel_event=None, stage_callback=None):
            seen.append(source.safe_filename)
            if source.safe_filename == "slow.wav":
                stage_callback("ASR")
                entered.set()
                assert release.wait(5)
                assert cancel_event.is_set()
            return Result(input_name=source.safe_filename)

    worker = begin(jobs, Blocking())
    try:
        assert entered.wait(4)
        assert jobs.cancel(slow.identifier)
        # A separate service/browser can observe the persisted intent.
        another_session = JobService(storage_root=root)
        pending = another_session.get_status(slow.identifier)
        assert pending.status == "IN_PROGRESS"
        assert pending.error_code == "CANCEL_REQUESTED"
        assert another_session.cancel(slow.identifier)
        assert jobs.get_status(next_job.identifier).status == "QUEUED"
    finally:
        release.set()
        assert worker.stop(timeout=5)

    cancelled = await_state(jobs, slow.identifier, "CANCELLED")
    assert cancelled.error_code is None
    assert cancelled.progress < 1
    assert cancelled.finished_at
    assert cancelled.attempt == 1
    with pytest.raises(JobIntakeError) as exc:
        jobs.get_artifact(slow.identifier, format="txt")
    assert exc.value.code == "ARTIFACT_NOT_READY"
    # stop was signaled before current ASR finished; follower stayed queued.
    assert jobs.get_status(next_job.identifier).status == "QUEUED"
    continued = begin(jobs, Blocking())
    try:
        await_state(jobs, next_job.identifier, "COMPLETED")
    finally:
        assert continued.stop(timeout=5)
    assert seen == ["slow.wav", "next.wav"]


def test_cancelled_stage_callback_does_not_publish_or_fail(env):
    jobs, root, add = env
    ticket = add("stages")
    entered, release = Event(), Event()

    class Staged:
        def transcribe(self, source, *, stage_callback=None, cancel_event=None):
            stage_callback("VAD")
            entered.set()
            assert release.wait(5)
            stage_callback("ASR")  # Reads persisted cancellation.
            pytest.fail("Cancellation should have interrupted stage callback")

    worker = begin(jobs, Staged())
    try:
        assert entered.wait(4)
        assert jobs.cancel(ticket.identifier)
    finally:
        release.set()
        assert worker.stop(timeout=5)

    state = await_state(jobs, ticket.identifier, "CANCELLED")
    assert state.progress < 1.0
    assert not list((root / ticket.identifier).glob("result/attempt-*"))


def test_cancel_during_model_initialization_prevents_transcription(env):
    jobs, _, add = env
    ticket = add("initializing")
    started, proceed = Event(), Event()
    executed = []

    class MockModel:
        def transcribe(self, source, *, cancel_event=None):
            executed.append(source.safe_filename)
            return Result(input_name=source.safe_filename)

    def factory():
        started.set()
        assert proceed.wait(5)
        return MockModel()

    worker = JobWorker(jobs, transcriber_factory=factory, poll_interval=0.01)
    worker.start()
    try:
        assert started.wait(4)
        assert jobs.cancel(ticket.identifier)
    finally:
        proceed.set()
        assert worker.stop(timeout=5)
    assert jobs.get_status(ticket.identifier).status == "CANCELLED"
    assert executed == []


def test_cancel_during_export_cannot_commit_completed(env, monkeypatch):
    jobs, root, add = env
    ticket = add("near_export")
    import modules.jobs.worker as worker_module
    original = worker_module.publish_result_set

    def publish_then_cancel(result, directory, *, attempt):
        published = original(result, directory, attempt=attempt)
        assert jobs.cancel(ticket.identifier)
        return published

    monkeypatch.setattr(worker_module, "publish_result_set", publish_then_cancel)
    seen = []

    class Model:
        def transcribe(self, source):
            seen.append(source.safe_filename)
            return Result(input_name=source.safe_filename)

    worker = begin(jobs, Model())
    try:
        await_state(jobs, ticket.identifier, "CANCELLED")
    finally:
        assert worker.stop(timeout=5)
    assert seen == ["near_export.wav"]
    state = jobs.get_status(ticket.identifier)
    assert state.error_code is None
    assert state.progress < 1.0
    assert (root / ticket.identifier / "result" / "attempt-1" / "transcript.txt").is_file()
    # Keep orphan bundle for explicit cleanup/recovery; never claim success.
    with closing(jobs._connect()) as conn:
        row = conn.execute(
            "SELECT output_json_path,output_txt_path FROM tasks WHERE uuid=?",
            (ticket.identifier,),
        ).fetchone()
    assert row["output_json_path"] is None
    assert row["output_txt_path"] is None


def test_crash_after_cancel_request_cannot_recover_output_as_completed(env):
    jobs, root, add = env
    ticket = add("crash_cancel")
    claimant = JobWorker(jobs)
    claim = claimant._claim()
    assert claim and claim.identifier == ticket.identifier
    assert jobs.cancel(ticket.identifier)
    publish_result_set(
        Result(input_name="crash_cancel.wav"), root / ticket.identifier,
        attempt=claim.attempt,
    )

    seen = []
    class Spy:
        def transcribe(self, source):
            seen.append(source.safe_filename)
            return Result(input_name=source.safe_filename)

    worker = begin(jobs, Spy())
    try:
        assert await_state(jobs, ticket.identifier, "CANCELLED")
    finally:
        assert worker.stop(timeout=5)
    assert seen == []
    assert jobs.get_status(ticket.identifier).progress < 1
    assert not jobs.retry(ticket.identifier)
    with pytest.raises(JobIntakeError):
        jobs.get_artifact(ticket.identifier, format="txt")


def test_ffprobe_failure_after_intake_does_not_poison_next_job(env, monkeypatch):
    jobs, _, add = env
    broken, next_job = add("broken"), add("healthy")
    import modules.jobs.inputs as inputs
    base = inputs._probe_audio

    def fail_stored(path, *, binary, timeout):
        if path.name.endswith("-broken.wav"):
            raise InputValidationError("FFPROBE_FAILED", "private exception context")
        return base(path, binary=binary, timeout=timeout)

    monkeypatch.setattr(inputs, "_probe_audio", fail_stored)
    seen = []
    class Model:
        def transcribe(self, source):
            seen.append(source.safe_filename)
            return Result(input_name=source.safe_filename)

    worker = begin(jobs, Model())
    try:
        a = await_state(jobs, broken.identifier, "FAILED")
        b = await_state(jobs, next_job.identifier, "COMPLETED")
    finally:
        assert worker.stop(timeout=5)
    assert a.error_code == "FFPROBE_FAILED"
    assert "private" not in str(a)
    assert b.status == "COMPLETED"
    assert seen == ["healthy.wav"]
    assert jobs.saved_input(broken.identifier).is_file()
    assert jobs.retry(broken.identifier)


def test_late_cancel_is_refused_after_completed(env):
    jobs, _, add = env
    item = add("complete")
    class Model:
        def transcribe(self, source):
            return Result(input_name=source.safe_filename)

    worker = begin(jobs, Model())
    try:
        await_state(jobs, item.identifier, "COMPLETED")
    finally:
        assert worker.stop(timeout=5)
    assert jobs.cancel(item.identifier) is False
    assert jobs.get_status(item.identifier).status == "COMPLETED"


def test_failed_and_cancelled_retries_dont_duplicate_success_artifacts(env):
    jobs, root, add = env
    item = add("fail_once")
    calls = []

    class Model:
        def transcribe(self, source, *, cancel_event=None):
            calls.append(source.safe_filename)
            if len(calls) == 1:
                raise MemoryError("Out of memory. secret")
            return Result(input_name=source.safe_filename)

    first = begin(jobs, Model())
    try:
        failed = await_state(jobs, item.identifier, "FAILED")
    finally:
        assert first.stop(timeout=5)
    assert failed.error_code == "PROCESSING_FAILED"
    assert jobs.retry(item.identifier)
    again = begin(jobs, Model())
    try:
        done = await_state(jobs, item.identifier, "COMPLETED")
    finally:
        assert again.stop(timeout=5)
    assert done.attempt == 2
    published = list((root / item.identifier / "result").glob("attempt-*"))
    assert [path.name for path in published] == ["attempt-2"]
    assert jobs.get_artifact(item.identifier, format="txt").is_file()

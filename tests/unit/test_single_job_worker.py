"""P15 / GitHub PR16: real local SQLite/thread/files, fake ASR and FFprobe.

The fast suite never downloads models, uses a browser, or reads private audio.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from threading import Event, Lock
import time
import wave

import pytest

from modules.jobs.inputs import InputSource
from modules.jobs.service import JobService
from modules.jobs.transcription_service import TranscriptionServiceError
from modules.jobs.worker import JobWorker, WorkerError

pytestmark = pytest.mark.fast


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 6400)


@dataclass
class _Segment:
    start: float = 0.0
    end: float = 0.35
    text: str = "Проверка"
    speaker: str = "SPEAKER_00"
    words: tuple = ()


@dataclass
class _Result:
    input_name: str
    segments: tuple = (_Segment(),)
    elapsed_seconds: float = 0.5
    model_size: str = "small"
    language: str = "ru"
    diarization_requested: bool = True


class _FakeASR:
    def __init__(self, call, *, wait=None, started=None):
        self._call = call
        self._wait = wait
        self._started = started

    def transcribe(self, source):
        if self._started is not None:
            self._started.set()
        if self._wait is not None and not self._wait.wait(timeout=5):
            raise RuntimeError("fake ASR test timed out")
        self._call(source.safe_filename)
        return _Result(input_name=source.safe_filename)


@pytest.fixture(autouse=True)
def stub_ffprobe(monkeypatch):
    monkeypatch.setattr(
        "modules.jobs.inputs._probe_audio",
        lambda path, *, binary, timeout: (0.4, "pcm_s16le"),
    )


@pytest.fixture
def env(tmp_path):
    source = tmp_path / "approved"
    source.mkdir()
    storage = tmp_path / "data" / "jobs"
    service = JobService(storage_root=storage, local_roots=(source,))
    def add(name):
        path = source / f"{name}.wav"
        _wav(path)
        return service.submit(InputSource.local_file(path))
    return service, storage, add


def _await_status(svc, identifier, status, *, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ticket = svc.get(identifier)
        if ticket is not None and ticket.status == status:
            return ticket
        time.sleep(0.01)
    pytest.fail(f"Job did not reach {status}; last={svc.get(identifier)}")


def _row(service, uuid):
    with closing(service._connect()) as conn:
        return conn.execute("""
            SELECT status, stage, attempt, error_code,
                   output_txt_path, output_json_path
            FROM tasks WHERE uuid=?
        """, (uuid,)).fetchone()


def test_fifo_single_owner_reconnect_and_artifacts(env):
    svc, storage, add = env
    tickets = [add(str(i)) for i in range(4)]
    calls = []
    concurrency = [0, 0]
    guard = Lock()
    factory_calls = [0]

    class ASR:
        def transcribe(self, source):
            with guard:
                concurrency[0] += 1
                concurrency[1] = max(concurrency[1], concurrency[0])
                calls.append(source.safe_filename)
            time.sleep(0.025)
            with guard:
                concurrency[0] -= 1
            return _Result(input_name=source.safe_filename)

    def factory():
        factory_calls[0] += 1
        return ASR()

    w = JobWorker(svc, transcriber_factory=factory, poll_interval=0.01)
    w.start()
    try:
        for ticket in tickets:
            _await_status(svc, ticket.identifier, "COMPLETED")
    finally:
        assert w.stop(wait=True, timeout=5)

    assert calls == [f"{n}.wav" for n in range(4)]
    assert concurrency[1] == 1
    assert factory_calls == [1]
    reopened = JobService(storage_root=storage)
    assert [x.identifier for x in reopened.list_jobs()] == [
        x.identifier for x in reversed(tickets)
    ]
    for ticket in tickets:
        row = _row(svc, ticket.identifier)
        assert row["attempt"] == 1
        assert row["stage"] == "DONE"
        assert (storage / row["output_txt_path"]).is_file()
        assert (storage / row["output_json_path"]).is_file()
        assert "SPEAKER_00" in (storage / row["output_txt_path"]).read_text("utf-8")


def test_first_failure_does_not_block_next_or_expose_exception(env):
    svc, _, add = env
    a, b = add("first"), add("second")
    order = []
    class ASR:
        def transcribe(self, source):
            order.append(source.safe_filename)
            if source.safe_filename == "first.wav":
                raise TranscriptionServiceError("DIARIZATION_MISSING", "private details")
            return _Result(input_name=source.safe_filename)

    worker = JobWorker(svc, transcriber_factory=ASR, poll_interval=0.01)
    worker.start()
    try:
        failed = _await_status(svc, a.identifier, "FAILED")
        _await_status(svc, b.identifier, "COMPLETED")
    finally:
        assert worker.stop(timeout=5)
    assert order == ["first.wav", "second.wav"]
    assert failed.error_code == "DIARIZATION_MISSING"
    assert "private details" not in str(failed)
    assert _row(svc, a.identifier)["output_txt_path"] is None


def test_explicit_retry_reuses_input_not_inference_on_request_thread(env):
    svc, storage, add = env
    a = add("retry")
    count = [0]
    class ASR:
        def transcribe(self, source):
            count[0] += 1
            if count[0] == 1:
                raise TranscriptionServiceError("PIPELINE_FAILED", "exception context")
            return _Result(input_name=source.safe_filename)
    worker = JobWorker(svc, transcriber_factory=ASR, poll_interval=0.01)
    worker.start()
    try:
        _await_status(svc, a.identifier, "FAILED")
        assert svc.retry(a.identifier)
        assert svc.get(a.identifier).status in ("QUEUED", "IN_PROGRESS", "COMPLETED")
        ok = _await_status(svc, a.identifier, "COMPLETED")
    finally:
        assert worker.stop(timeout=5)
    assert ok.attempt == 2
    assert (storage / a.identifier / "result" / "transcript-2.txt").is_file()
    assert not svc.retry(a.identifier)


def test_cancel_only_queued_never_calls_model(env):
    svc, _, add = env
    a, b = add("cancel"), add("keep")
    assert svc.cancel(a.identifier)
    assert not svc.cancel(a.identifier)
    assert svc.get(a.identifier).status == "CANCELLED"
    calls = []
    worker = JobWorker(svc, transcriber_factory=lambda: _FakeASR(calls.append),
                       poll_interval=0.01)
    worker.start()
    try:
        _await_status(svc, b.identifier, "COMPLETED")
    finally:
        assert worker.stop(timeout=5)
    assert calls == ["keep.wav"]
    assert not svc.retry(a.identifier)


def test_refuses_duplicate_worker_even_in_same_process(env):
    svc, _, add = env
    a = add("single")
    gate = Event()
    begun = Event()
    first = JobWorker(
        svc, transcriber_factory=lambda: _FakeASR(lambda _: None, wait=gate, started=begun),
        poll_interval=0.01,
    )
    second = JobWorker(svc, transcriber_factory=lambda: _FakeASR(lambda _: None),
                       poll_interval=0.01)
    first.start()
    try:
        assert begun.wait(4)
        with pytest.raises(WorkerError) as exc:
            second.start()
        assert exc.value.code == "WORKER_BUSY"
        with pytest.raises(WorkerError) as twice:
            first.start()
        assert twice.value.code == "WORKER_BUSY"
    finally:
        gate.set()
        assert first.stop(timeout=5)
    assert not second.is_running
    assert svc.get(a.identifier).status == "COMPLETED"


def test_stop_does_not_abort_current_job_or_start_next(env):
    svc, _, add = env
    current, later = add("current"), add("later")
    entered, release = Event(), Event()
    calls = []
    worker = JobWorker(svc, transcriber_factory=lambda: _FakeASR(
        calls.append, started=entered, wait=release), poll_interval=0.01)
    worker.start()
    try:
        assert entered.wait(4)
        assert not worker.stop(wait=False)
        assert svc.get(later.identifier).status == "QUEUED"
    finally:
        release.set()
        assert worker.stop(wait=True, timeout=5)
    assert svc.get(current.identifier).status == "COMPLETED"
    assert svc.get(later.identifier).status == "QUEUED"
    next_worker = JobWorker(
        svc, transcriber_factory=lambda: _FakeASR(calls.append), poll_interval=0.01,
    )
    next_worker.start()
    try:
        _await_status(svc, later.identifier, "COMPLETED")
    finally:
        assert next_worker.stop(timeout=5)
    assert calls == ["current.wav", "later.wav"]


def test_stale_in_progress_requires_recovery_no_false_completion(env):
    svc, _, add = env
    stale, waiting = add("stale"), add("pending")
    with closing(svc._connect()) as conn:
        with conn:
            conn.execute(
                "UPDATE tasks SET status='IN_PROGRESS' WHERE uuid=?",
                (stale.identifier,),
            )
    worker = JobWorker(svc, transcriber_factory=lambda: _FakeASR(lambda _: None))
    with pytest.raises(WorkerError) as e:
        worker.start()
    assert e.value.code == "RECOVERY_REQUIRED"
    assert not worker.is_running
    assert svc.get(waiting.identifier).status == "QUEUED"


def test_missing_stored_input_fails_and_follower_completes(env):
    svc, _, add = env
    missing, good = add("gone"), add("good")
    svc.saved_input(missing.identifier).unlink()
    worker = JobWorker(svc, transcriber_factory=lambda: _FakeASR(lambda _: None),
                       poll_interval=0.01)
    worker.start()
    try:
        failed = _await_status(svc, missing.identifier, "FAILED")
        _await_status(svc, good.identifier, "COMPLETED")
    finally:
        assert worker.stop(timeout=5)
    assert failed.error_code == "STORED_INPUT_MISSING"
    assert not svc.retry(missing.identifier)


def test_factory_failure_marks_task_failed_then_next_model_succeeds(env):
    svc, _, add = env
    first, next_ = add("first"), add("next")
    attempts = [0]
    def factory():
        attempts[0] += 1
        if attempts[0] == 1:
            raise RuntimeError("no token, hide me")
        return _FakeASR(lambda _: None)
    worker = JobWorker(svc, transcriber_factory=factory, poll_interval=0.01)
    worker.start()
    try:
        assert _await_status(svc, first.identifier, "FAILED").error_code == "MODEL_INIT_FAILED"
        _await_status(svc, next_.identifier, "COMPLETED")
    finally:
        assert worker.stop(timeout=5)
    assert attempts == [2]


def test_result_format_error_never_marks_completed(env):
    svc, storage, add = env
    job = add("invalid")
    class Bad:
        def transcribe(self, source):
            return _Result(input_name=source.safe_filename, segments=())
    w = JobWorker(svc, transcriber_factory=Bad, poll_interval=0.01)
    w.start()
    try:
        failed = _await_status(svc, job.identifier, "FAILED")
    finally:
        assert w.stop(timeout=5)
    assert failed.error_code == "EMPTY_TRANSCRIPT"
    assert _row(svc, job.identifier)["output_json_path"] is None
    assert svc.saved_input(job.identifier).is_file()


def test_symlink_result_directory_is_refused_before_export(env):
    svc, storage, add = env
    job = add("symlink")
    outside = storage.parent.parent / "outside-results"
    outside.mkdir()
    folder = storage / job.identifier / "result"
    try:
        folder.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable")
    worker = JobWorker(
        svc, transcriber_factory=lambda: _FakeASR(lambda _: None),
        poll_interval=0.01,
    )
    worker.start()
    try:
        failed = _await_status(svc, job.identifier, "FAILED")
    finally:
        assert worker.stop(timeout=5)
    assert failed.error_code == "UNSAFE_RESULT_DIR"
    assert list(outside.iterdir()) == []


def test_arbitrary_exception_code_is_not_persisted(env):
    svc, _, add = env
    job = add("sensitive")
    class Bad:
        def transcribe(self, source):
            exc = RuntimeError("HF token secret should not appear")
            exc.code = "SENSITIVE_TOKEN_VALUE"
            raise exc
    worker = JobWorker(svc, transcriber_factory=Bad, poll_interval=0.01)
    worker.start()
    try:
        failed = _await_status(svc, job.identifier, "FAILED")
    finally:
        assert worker.stop(timeout=5)
    assert failed.error_code == "PROCESSING_FAILED"

"""PR19/P18: real SQLite, files, worker lock and recovery without models."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Event
import sqlite3
import time
import wave
from uuid import uuid4

import pytest

from modules.jobs.artifacts import publish_result_set
from modules.jobs.inputs import InputSource
from modules.jobs.recovery import RecoveryError
from modules.jobs.service import JobService
from modules.jobs.worker import JobWorker, WorkerError

pytestmark = pytest.mark.fast


@dataclass
class Segment:
    start: float = 0.0
    end: float = 0.35
    text: str = "Проверка"
    speaker: str = "SPEAKER_00"
    words: tuple = ()


@dataclass
class Result:
    input_name: str = "example.wav"
    segments: tuple = (Segment(),)
    elapsed_seconds: float = 0.5
    model_size: str = "small"
    language: str = "ru"
    diarization_requested: bool = True


class FakeASR:
    def __init__(self, seen):
        self.seen = seen

    def transcribe(self, source):
        self.seen.append(source.safe_filename)
        return Result(input_name=source.safe_filename)


@pytest.fixture(autouse=True)
def fake_ffprobe(monkeypatch):
    monkeypatch.setattr(
        "modules.jobs.inputs._probe_audio",
        lambda path, *, binary, timeout: (0.4, "pcm_s16le"),
    )


@pytest.fixture
def env(tmp_path):
    src = tmp_path / "approved"
    src.mkdir()
    root = tmp_path / "data" / "jobs"
    svc = JobService(storage_root=root, local_roots=(src,))

    def add(name):
        path = src / (name + ".wav")
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 6400)
        return svc.submit(InputSource.local_file(path))

    return svc, root, add


def await_status(svc, identifier, status):
    end = time.monotonic() + 6
    while time.monotonic() < end:
        ticket = svc.get_status(identifier)
        if ticket and ticket.status == status:
            return ticket
        time.sleep(0.01)
    pytest.fail("Expected persisted job state was not reached")


def simulate_crash_claim(svc, identifier, *, milestone="ALIGN"):
    """Claim like a worker that is killed without running its main thread."""
    probe = JobWorker(svc)
    claim = probe._claim()
    assert claim is not None and claim.identifier == identifier
    assert probe._record_stage(claim, milestone)
    return claim


def start_with_fake(svc, seen):
    worker = JobWorker(
        svc, transcriber_factory=lambda: FakeASR(seen), poll_interval=0.01,
    )
    worker.start()
    return worker


def test_interrupted_is_not_auto_reprocessed_and_waiting_queue_continues(env):
    svc, root, add = env
    first, waiting = add("unfinished"), add("waiting")
    simulate_crash_claim(svc, first.identifier)
    seen = []
    worker = start_with_fake(svc, seen)
    try:
        completed = await_status(svc, waiting.identifier, "COMPLETED")
    finally:
        assert worker.stop(timeout=5)
    state = svc.get_status(first.identifier)
    assert state.status == "INTERRUPTED"
    assert state.stage == "INTERRUPTED"
    assert state.error_code == "RECOVERY_INTERRUPTED"
    assert state.progress == 0.72  # Last actual ALIGN milestone, not a fake 100%.
    assert state.finished_at is not None
    assert worker.last_recovery.interrupted == 1
    assert worker.last_recovery.queued_preserved == 1
    assert worker.last_recovery.completed_from_artifacts == 0
    assert seen == ["waiting.wav"]
    assert completed.progress == 1.0
    assert svc.saved_input(first.identifier).is_file()
    assert svc.get_artifact(waiting.identifier, format="txt").is_file()


def test_valid_published_bundle_finishes_without_second_inference(env):
    svc, root, add = env
    ticket = add("published")
    claim = simulate_crash_claim(svc, ticket.identifier, milestone="EXPORT")
    bundle = publish_result_set(Result(input_name="published.wav"),
                                root / ticket.identifier, attempt=claim.attempt)
    json_before = bundle.files.json_path.read_bytes()
    seen = []
    worker = start_with_fake(svc, seen)
    try:
        recovered = await_status(svc, ticket.identifier, "COMPLETED")
    finally:
        assert worker.stop(timeout=5)
    assert seen == []
    assert worker.last_recovery.completed_from_artifacts == 1
    assert worker.last_recovery.interrupted == 0
    assert recovered.attempt == 1 and recovered.progress == 1
    assert recovered.stage == "DONE" and recovered.finished_at is not None
    assert svc.get_artifact(ticket.identifier, format="json").read_bytes() == json_before
    assert svc.get_artifact(ticket.identifier, format="txt").is_file()


def test_invalid_current_attempt_bundle_remains_interrupted_no_fake_success(env):
    svc, root, add = env
    ticket = add("corrupt")
    claim = simulate_crash_claim(svc, ticket.identifier, milestone="EXPORT")
    bundle = publish_result_set(Result(), root / ticket.identifier, attempt=claim.attempt)
    bundle.files.txt_path.write_bytes(b"corrupted")
    worker = start_with_fake(svc, [])
    try:
        item = await_status(svc, ticket.identifier, "INTERRUPTED")
    finally:
        assert worker.stop(timeout=5)
    assert item.error_code == "RECOVERY_OUTPUT_UNVERIFIED"
    assert item.progress == 0.95
    assert worker.last_recovery.unverified_outputs == 1
    assert bundle.files.txt_path.read_bytes() == b"corrupted"
    assert svc.saved_input(ticket.identifier).is_file()


def test_recovery_preserves_pending_and_orphan_files_without_following_symlinks(env):
    svc, root, add = env
    ticket = add("unfinished")
    simulate_crash_claim(svc, ticket.identifier)
    result_root = root / ticket.identifier / "result"
    result_root.mkdir()
    temp = result_root / (".attempt-1-pending-" + "a" * 32)
    temp.mkdir()
    (temp / "transcript.json").write_text("incomplete", encoding="utf-8")
    orphan = root / str(uuid4())
    orphan.mkdir()
    (orphan / "input").mkdir()
    (orphan / "input" / "record.wav").write_bytes(b"private audio")
    worker = start_with_fake(svc, [])
    try:
        state = await_status(svc, ticket.identifier, "INTERRUPTED")
    finally:
        assert worker.stop(timeout=5)
    assert state.status == "INTERRUPTED"
    report = worker.last_recovery
    assert report.orphan_job_directories == 1
    assert report.unfinished_output_directories == 1
    assert temp.joinpath("transcript.json").read_text() == "incomplete"
    assert orphan.joinpath("input", "record.wav").read_bytes() == b"private audio"
    assert svc.get(str(orphan.name)) is None


def test_explicit_retry_increments_attempt_and_preserves_old_bundle(env):
    svc, root, add = env
    ticket = add("retry")
    attempt_one = simulate_crash_claim(svc, ticket.identifier, milestone="DIARIZE")
    # A surviving earlier (failed) output is not enough to complete attempt 2.
    old = publish_result_set(Result(), root / ticket.identifier, attempt=1)
    original_data = old.files.txt_path.read_bytes()
    with closing(svc._connect()) as conn:
        with conn:
            conn.execute("""
                UPDATE tasks SET attempt=2, stage='ASR', progress=0.2
                WHERE uuid=?
            """, (ticket.identifier,))
    worker = start_with_fake(svc, [])
    try:
        interrupted = await_status(svc, ticket.identifier, "INTERRUPTED")
    finally:
        assert worker.stop(timeout=5)
    assert interrupted.attempt == 2
    assert interrupted.error_code == "RECOVERY_INTERRUPTED"
    assert old.files.txt_path.read_bytes() == original_data
    assert svc.retry(ticket.identifier)
    reset = svc.get_status(ticket.identifier)
    assert reset.status == "QUEUED" and reset.attempt == 2
    assert reset.progress == 0 and reset.started_at is None
    assert reset.finished_at is None and reset.error_code is None
    seen = []
    resumed = start_with_fake(svc, seen)
    try:
        final = await_status(svc, ticket.identifier, "COMPLETED")
    finally:
        assert resumed.stop(timeout=5)
    assert final.attempt == 3
    assert seen == ["retry.wav"]
    assert svc.get_artifact(ticket.identifier, format="txt").parent.name == "attempt-3"
    assert old.files.txt_path.read_bytes() == original_data


def test_retry_interrupted_with_missing_input_is_refused(env):
    svc, _, add = env
    ticket = add("missing")
    simulate_crash_claim(svc, ticket.identifier)
    worker = start_with_fake(svc, [])
    try:
        await_status(svc, ticket.identifier, "INTERRUPTED")
    finally:
        assert worker.stop(timeout=5)
    svc.saved_input(ticket.identifier).unlink()
    assert not svc.retry(ticket.identifier)
    assert svc.get(ticket.identifier).status == "INTERRUPTED"


def test_second_start_is_idempotent_no_rewrite(env):
    svc, root, add = env
    ticket = add("repeat")
    simulate_crash_claim(svc, ticket.identifier)
    first = start_with_fake(svc, [])
    try:
        state = await_status(svc, ticket.identifier, "INTERRUPTED")
    finally:
        assert first.stop(timeout=5)
    assert first.last_recovery.interrupted == 1
    reopened = JobService(storage_root=root)
    second = start_with_fake(reopened, [])
    try:
        second_state = reopened.get(ticket.identifier)
    finally:
        assert second.stop(timeout=5)
    assert second.last_recovery.interrupted == 0
    assert second.last_recovery.completed_from_artifacts == 0
    assert second_state == state


def test_running_worker_lock_blocks_other_worker_before_reconciliation(env):
    svc, _, add = env
    first = add("busy")
    seen = []
    started, release = Event(), Event()

    class Blocking:
        def transcribe(self, source):
            started.set()
            assert release.wait(5)
            seen.append(source.safe_filename)
            return Result(input_name=source.safe_filename)

    owner = JobWorker(svc, transcriber_factory=Blocking, poll_interval=0.01)
    competing = JobWorker(svc, transcriber_factory=lambda: FakeASR(seen))
    owner.start()
    try:
        assert started.wait(4)
        with pytest.raises(WorkerError) as exc:
            competing.start()
        assert exc.value.code == "WORKER_BUSY"
        assert svc.get(first.identifier).status == "IN_PROGRESS"
        assert competing.last_recovery is None
    finally:
        release.set()
        assert owner.stop(timeout=5)
    assert svc.get(first.identifier).status == "COMPLETED"


def test_sqlite_error_rolls_back_all_recovery_updates_and_releases_lock(env):
    svc, _, add = env
    first, second = add("first"), add("second")
    simulate_crash_claim(svc, first.identifier)
    with closing(svc._connect()) as conn:
        with conn:
            conn.execute("""
                UPDATE tasks SET status='IN_PROGRESS', attempt=1
                WHERE uuid=?
            """, (second.identifier,))
            conn.execute("""
                CREATE TRIGGER reject_recovery BEFORE UPDATE ON tasks
                WHEN NEW.status='INTERRUPTED' AND OLD.uuid=?
                BEGIN SELECT RAISE(FAIL, 'simulated recovery failure'); END
            """.replace("OLD.uuid=?", "OLD.uuid='" + second.identifier + "'"))
    faulty = JobWorker(svc, transcriber_factory=lambda: FakeASR([]))
    with pytest.raises(sqlite3.DatabaseError):
        faulty.start()
    assert not faulty.is_running
    assert faulty.last_recovery is None
    assert svc.get(first.identifier).status == "IN_PROGRESS"
    assert svc.get(second.identifier).status == "IN_PROGRESS"
    with closing(svc._connect()) as conn:
        with conn:
            conn.execute("DROP TRIGGER reject_recovery")
    safe = start_with_fake(svc, [])
    try:
        await_status(svc, first.identifier, "INTERRUPTED")
        await_status(svc, second.identifier, "INTERRUPTED")
    finally:
        assert safe.stop(timeout=5)
    assert safe.last_recovery.interrupted == 2


def test_completed_failed_and_cancelled_rows_are_never_mutated(env):
    svc, _, add = env
    done = add("done")
    cancelled = add("cancelled")
    fail = add("failed")
    assert svc.cancel(cancelled.identifier)
    with closing(svc._connect()) as conn:
        with conn:
            conn.execute("UPDATE tasks SET status='FAILED', error_code='ASR_FAILED' WHERE uuid=?",
                         (fail.identifier,))
            conn.execute("UPDATE tasks SET status='COMPLETED', stage='DONE', progress=1.0 WHERE uuid=?",
                         (done.identifier,))
    before = {x.identifier: x for x in svc.list_jobs()}
    worker = start_with_fake(svc, [])
    try:
        assert worker.last_recovery.queued_preserved == 0
    finally:
        assert worker.stop(timeout=5)
    after = {x.identifier: x for x in svc.list_jobs()}
    assert after == before

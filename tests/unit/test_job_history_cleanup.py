"""P24: SQLite-backed history, verified previews, and confirmed local deletion.

Tests run offline with actual SQLite/files, a fake ASR and no Gradio/ML import.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4
import wave

import pytest

from modules.jobs.inputs import InputSource
from modules.jobs.service import JobIntakeError, JobService
from modules.jobs.worker import JobWorker
from modules.ui.simple_mode import SimpleModeController, _PREVIEW_BYTES

pytestmark = pytest.mark.fast


@dataclass
class Segment:
    start: float = 12.0
    end: float = 14.0
    speaker: str = "SPEAKER_00"
    text: str = "SPEAKER_00|Добрый день"


@dataclass
class Result:
    segments: tuple = (Segment(),)
    input_name: str = "record.wav"
    elapsed_seconds: float = 1.0
    model_size: str = "small"
    language: str = "ru"
    diarization_requested: bool = True


class FakeASR:
    def transcribe(self, source):
        return Result(input_name=source.safe_filename)


@pytest.fixture(autouse=True)
def ffprobe(monkeypatch):
    monkeypatch.setattr(
        "modules.jobs.inputs._probe_audio",
        lambda path, *, binary, timeout: (1.0, "pcm_s16le"),
    )


@pytest.fixture
def env(tmp_path):
    source_dir = tmp_path / "approved"
    source_dir.mkdir()
    jobs = JobService(storage_root=tmp_path / "jobs", local_roots=(source_dir,))
    def add(name="record"):
        file = source_dir / (name + ".wav")
        with wave.open(str(file), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\x00\x00" * 2000)
        return jobs.submit(InputSource.local_file(file))
    def complete(identifier):
        worker = JobWorker(jobs, transcriber_factory=FakeASR, poll_interval=0.01)
        worker.start()
        try:
            until = time.monotonic() + 5
            while time.monotonic() < until:
                status = jobs.get_status(identifier)
                if status.status in ("COMPLETED", "FAILED"):
                    break
                time.sleep(0.01)
            assert jobs.get(identifier).status == "COMPLETED"
        finally:
            assert worker.stop(timeout=5)
    class IdleOwner:
        def __init__(self):
            self.jobs = jobs
            self.is_running = True
            self.wakes = 0
        def wake(self):
            self.wakes += 1
    controller = SimpleModeController(jobs, IdleOwner())
    return jobs, controller, add, complete


def test_recent_history_persists_after_recreating_service_and_selects_completed(env):
    jobs, controller, add, complete = env
    early, latest = add("first"), add("second")
    complete(latest.identifier)
    copy = JobService(storage_root=jobs.storage.root)
    fresh = SimpleModeController(copy, type("Owner", (), {"jobs":copy, "is_running":True})())
    choices, selected = fresh.history_options(latest.identifier)
    assert [u for _, u in choices][:2] == [latest.identifier, early.identifier]
    assert selected == latest.identifier
    assert all("small" in label and "QUEUED" in label or "COMPLETED" in label for label, _ in choices)
    assert fresh.history_options(str(uuid4()))[1] is None
    data = fresh.select_history(latest.identifier)
    assert data[0] == latest.identifier
    assert data[4] == str(copy.get_artifact(latest.identifier, format="txt"))
    assert "[SPEAKER_00]: Добрый день" in data[6]
    assert data[7] is False
    assert copy.get_status(latest.identifier).status == "COMPLETED"


def test_preview_is_bounded_and_only_uses_verified_txt(env):
    jobs, controller, add, complete = env
    task = add()
    assert controller.preview(task.identifier) == ""
    complete(task.identifier)
    path = jobs.get_artifact(task.identifier, format="txt")
    assert controller.preview(task.identifier) == path.read_text("utf-8")
    # An altered bundle cannot be previewed.
    path.write_text("tampered secret", encoding="utf-8")
    assert controller.preview(task.identifier) == ""
    assert controller.poll_with_preview(task.identifier)[-1] == ""
    # Ensure preview reader never calls read() without an upper bound.
    import inspect
    body = inspect.getsource(SimpleModeController.preview)
    assert "read(_PREVIEW_BYTES + 1)" in body
    assert _PREVIEW_BYTES <= 16 * 1024


def test_open_folder_only_after_verified_completion(env):
    jobs, controller, add, complete = env
    task = add()
    opened = []
    assert "только" in controller.open_result_folder(task.identifier, opener=opened.append)
    assert opened == []
    complete(task.identifier)
    assert "открыта" in controller.open_result_folder(task.identifier, opener=opened.append)
    assert opened == [str(jobs.get_artifact(task.identifier, format="txt").parent)]


def test_confirmed_delete_removes_selected_terminal_only_and_history(env):
    jobs, controller, add, complete = env
    deleted = add("delete")
    untouched = add("keep")
    complete(deleted.identifier)
    primary = jobs.saved_input(deleted.identifier)
    output = jobs.get_artifact(deleted.identifier, format="txt")
    assert output.exists() and primary.exists()

    reject, saved, checked = controller.delete_selected(
        deleted.identifier, False, deleted.identifier,
    )
    assert "флажок" in reject and saved == deleted.identifier and checked is False
    assert output.exists()
    done, stored_id, checked = controller.delete_selected(
        deleted.identifier, True, deleted.identifier,
    )
    assert "удалены" in done and stored_id == "" and checked is False
    assert not output.exists() and not primary.exists()
    assert jobs.get_status(deleted.identifier) is None
    assert not (jobs.storage.root / deleted.identifier).exists()
    assert jobs.get_status(untouched.identifier).status == "QUEUED"
    assert jobs.saved_input(untouched.identifier).exists()
    assert controller.history_options("")[0][0][1] == untouched.identifier


@pytest.mark.parametrize("state", ["QUEUED", "IN_PROGRESS"])
def test_active_or_queued_delete_is_refused_under_database_lock(env, state):
    jobs, _, add, _ = env
    t = add()
    with closing(jobs._connect()) as conn:
        with conn:
            conn.execute("UPDATE tasks SET status=? WHERE uuid=?", (state, t.identifier))
    with pytest.raises(JobIntakeError) as error:
        jobs.delete_terminal(t.identifier, confirmed=True)
    assert error.value.code == "JOB_ACTIVE"
    assert jobs.get_status(t.identifier) is not None
    assert jobs.saved_input(t.identifier).exists()


def test_invalid_uuid_missing_uuid_and_no_confirm_cannot_delete(env):
    jobs, _, add, complete = env
    t = add()
    complete(t.identifier)
    for u in (str(uuid4()), "../secret", "", None):
        with pytest.raises(JobIntakeError) as err:
            jobs.delete_terminal(u, confirmed=True)
        assert err.value.code == "JOB_NOT_FOUND"
    with pytest.raises(JobIntakeError) as err:
        jobs.delete_terminal(t.identifier)
    assert err.value.code == "DELETE_NOT_CONFIRMED"
    assert jobs.get_artifact(t.identifier, format="txt").exists()


def test_sqlite_delete_failure_restores_renamed_folder_and_row(env):
    jobs, _, add, complete = env
    t = add()
    complete(t.identifier)
    with closing(jobs._connect()) as conn:
        with conn:
            conn.execute("""
                CREATE TRIGGER forbid_delete BEFORE DELETE ON tasks
                BEGIN SELECT RAISE(ABORT, 'blocked by test'); END
            """)
    with pytest.raises(sqlite3.IntegrityError):
        jobs.delete_terminal(t.identifier, confirmed=True)
    assert (jobs.storage.root / t.identifier).is_dir()
    assert not (jobs.storage.root / f".deleting-{t.identifier}").exists()
    assert jobs.get_status(t.identifier).status == "COMPLETED"
    assert jobs.get_artifact(t.identifier, format="txt").exists()


def test_cleanup_failure_reports_residual_private_bytes(env, monkeypatch):
    jobs, _, add, complete = env
    t = add()
    complete(t.identifier)
    def fail_cleanup(path):
        raise OSError("private system location")
    monkeypatch.setattr("modules.jobs.service.shutil.rmtree", fail_cleanup)
    with pytest.raises(JobIntakeError) as err:
        jobs.delete_terminal(t.identifier, confirmed=True)
    assert err.value.code == "DELETE_CLEANUP_PENDING"
    assert "private system location" not in str(err.value)
    assert jobs.get_status(t.identifier) is None
    assert (jobs.storage.root / f".deleting-{t.identifier}").is_dir()


def test_symlinked_job_cannot_delete_outside_directory(env, tmp_path):
    jobs, _, add, complete = env
    t = add()
    complete(t.identifier)
    folder = jobs.storage.root / t.identifier
    outside = tmp_path / "unrelated"
    outside.mkdir()
    (outside / "keep.txt").write_text("private", encoding="utf-8")
    moved = tmp_path / "owned"
    folder.rename(moved)
    try:
        folder.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        moved.rename(folder)
        pytest.skip("Symlinks are not permitted on this host")
    with pytest.raises(JobIntakeError) as err:
        jobs.delete_terminal(t.identifier, confirmed=True)
    assert err.value.code == "UNSAFE_JOB_DIRECTORY"
    assert (outside / "keep.txt").read_text("utf-8") == "private"
    folder.unlink()
    moved.rename(folder)

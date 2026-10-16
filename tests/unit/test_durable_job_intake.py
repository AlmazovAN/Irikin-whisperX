"""PR15 / P14: offline durable audio intake, restart and failure safety.

No FFmpeg, Torch, network or personal recordings required by fast CI.
"""
from __future__ import annotations

import sqlite3
import wave
from pathlib import Path
from uuid import uuid4

import pytest

from modules.jobs.inputs import (
    InputLimits, InputSource, InputValidationError, validate_input,
)
from modules.jobs.service import JobIntakeError, JobService
from modules.jobs.storage import JobStorage, JobStorageError

pytestmark = pytest.mark.fast


@pytest.fixture(autouse=True)
def mock_audio_probe(monkeypatch):
    """Mock only FFprobe; exercise actual validation, byte copy and SQLite."""
    monkeypatch.setattr(
        "modules.jobs.inputs._probe_audio",
        lambda path, *, binary, timeout: (0.4, "pcm_s16le"),
    )


@pytest.fixture
def setup(tmp_path):
    cache = tmp_path / "gradio-cache"
    cache.mkdir()
    local = tmp_path / "allowed"
    local.mkdir()
    jobs = tmp_path / "data" / "jobs"
    return cache, local, jobs


def wav(path: Path) -> bytes:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 6400)
    return path.read_bytes()


def build_service(setup):
    cache, local, jobs = setup
    return JobService(storage_root=jobs, upload_roots=[cache], local_roots=[local])


def db_rows(jobs):
    with sqlite3.connect(jobs / "jobs.sqlite3") as db:
        return db.execute(
            "SELECT uuid, status, stage, input_path, file_name, model_name,"
            " language, task_type, task_params FROM tasks ORDER BY id"
        ).fetchall()


def test_upload_kept_after_staging_deleted_and_new_service_started(setup):
    cache, _, jobs = setup
    file = cache / "gradio-file.wav"
    original = wav(file)
    svc = build_service(setup)
    ticket = svc.submit(InputSource.upload(file, original_name="Собрание.wav"))
    assert ticket.status == "QUEUED" and ticket.stage == "QUEUED"
    assert ticket.file_name == "Собрание.wav"
    assert ticket.model_name == "small" and ticket.language == "ru"
    persisted = svc.saved_input(ticket.identifier)
    assert persisted.read_bytes() == original
    assert persisted.relative_to(jobs).parts[1] == "input"

    file.unlink()  # Simulate the Gradio temporary cache disappearing.
    reopened = build_service(setup)
    assert reopened.get(ticket.identifier) == ticket
    assert reopened.saved_input(ticket.identifier).read_bytes() == original
    rows = db_rows(jobs)
    assert len(rows) == 1
    assert rows[0][1:3] == ("QUEUED", "QUEUED")
    assert rows[0][3].startswith(f"{ticket.identifier}/input/")
    assert "HF_TOKEN" not in rows[0][8]
    assert "transcribe" in rows[0][8]


def test_same_uploaded_display_name_creates_independent_jobs(setup):
    cache, _, jobs = setup
    first = cache / "gradio-one.wav"
    second = cache / "gradio-two.wav"
    wav(first)
    wav(second)
    svc = build_service(setup)
    one = svc.submit(InputSource.upload(first, original_name="same.wav"))
    two = svc.submit(InputSource.upload(second, original_name="same.wav"))
    assert one.identifier != two.identifier
    assert svc.saved_input(one.identifier) != svc.saved_input(two.identifier)
    assert len(db_rows(jobs)) == 2
    assert svc.saved_input(one.identifier).is_file()


def test_local_authorized_input_also_works(setup):
    _, local, _ = setup
    sound = local / "ok.wav"
    wav(sound)
    ticket = build_service(setup).submit(InputSource.local_file(sound))
    assert ticket.file_name == "ok.wav"


def test_reject_unapproved_path_without_db_record(setup):
    cache, _, jobs = setup
    outside = cache.parent / "outside.wav"
    wav(outside)
    svc = build_service(setup)
    with pytest.raises(InputValidationError) as error:
        svc.submit(InputSource.local_file(outside))
    assert error.value.code == "ACCESS_DENIED"
    assert db_rows(jobs) == []
    assert list(jobs.glob("*/input/*")) == []


def test_reject_inaccessible_drive_source_without_job(setup):
    _, _, jobs = setup
    svc = build_service(setup)
    with pytest.raises(InputValidationError) as error:
        svc.submit(InputSource.google_drive_url("https://drive.google.com/file/d/fake"))
    assert error.value.code == "SOURCE_NOT_SUPPORTED"
    assert db_rows(jobs) == []


def test_reject_changed_file_after_p08_validation(setup):
    cache, _, jobs = setup
    original = cache / "audio.wav"
    wav(original)
    descriptor = validate_input(InputSource.upload(original), upload_roots=[cache])
    original.unlink()
    replacement = cache.parent / "private.wav"
    wav(replacement)
    try:
        original.symlink_to(replacement)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable")
    store = JobStorage(jobs)
    with pytest.raises(JobStorageError) as error:
        store.save_validated(str(uuid4()), descriptor, limits=InputLimits())
    assert error.value.code == "INPUT_CHANGED"
    assert list(jobs.glob("*/input/*")) == []


def test_copy_failure_leaves_no_part_or_queued_row(setup, monkeypatch):
    cache, _, jobs = setup
    f = cache / "upload.wav"
    wav(f)
    svc = build_service(setup)

    def fail_atomic_replace(*args):
        raise OSError("simulated storage failure")
    monkeypatch.setattr("modules.jobs.storage.os.replace", fail_atomic_replace)
    with pytest.raises(JobStorageError) as error:
        svc.submit(InputSource.upload(f))
    assert error.value.code == "STORAGE_FAILED"
    assert db_rows(jobs) == []
    assert list(jobs.iterdir()) == [jobs / "jobs.sqlite3"]


def test_sqlite_commit_error_removes_only_new_folder(setup, monkeypatch):
    cache, _, jobs = setup
    f = cache / "upload.wav"
    original = wav(f)
    svc = build_service(setup)
    existing = svc.submit(InputSource.upload(f))

    def fail_db(*args):
        raise sqlite3.OperationalError("disk unavailable")
    monkeypatch.setattr(svc, "_insert_queued", fail_db)
    with pytest.raises(JobIntakeError) as error:
        svc.submit(InputSource.upload(f))
    assert error.value.code == "QUEUE_PERSIST_FAILED"
    assert len(db_rows(jobs)) == 1
    assert svc.saved_input(existing.identifier).read_bytes() == original
    assert len([x for x in jobs.iterdir() if x.is_dir()]) == 1


def test_reprobe_failure_does_not_queue_invalid_copy(setup, monkeypatch):
    cache, _, jobs = setup
    file = cache / "ok.wav"
    wav(file)
    svc = build_service(setup)
    calls = 0

    def probe(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InputValidationError("INVALID_MEDIA", "Stored audio invalid")
        return 0.4, "pcm_s16le"
    monkeypatch.setattr("modules.jobs.inputs._probe_audio", probe)
    with pytest.raises(InputValidationError):
        svc.submit(InputSource.upload(file))
    assert calls == 2
    assert db_rows(jobs) == []
    assert len([x for x in jobs.iterdir() if x.is_dir()]) == 0


def test_no_private_path_or_secret_in_public_receipt(setup):
    cache, _, jobs = setup
    sound = cache / "upload.wav"
    wav(sound)
    svc = build_service(setup)
    t = svc.submit(InputSource.upload(sound))
    assert "data/jobs" not in repr(t)
    assert "input_path" not in repr(t)
    assert "hf_token" not in repr(t).lower()
    assert svc.get("not-an-id") is None
    assert svc.get(str(uuid4())) is None
    assert jobs.is_dir()


@pytest.mark.parametrize("path", [
    "../../other.wav", "/etc/passwd", "broken/input/file.wav",
])
def test_stored_locator_never_escapes_job_root(setup, path):
    svc = build_service(setup)
    with pytest.raises(JobStorageError):
        svc.storage.resolve_input(path)


def test_reject_symlink_storage_root(setup):
    cache, _, jobs = setup
    target = cache / "target-folder"
    target.mkdir()
    try:
        jobs.parent.mkdir(parents=True, exist_ok=True)
        jobs.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable")
    with pytest.raises(JobStorageError, match="symlink"):
        JobService(storage_root=jobs)


def test_actual_sqlite_insert_failure_rolls_back_new_job(setup):
    cache, _, jobs = setup
    source = cache / "reject.wav"
    wav(source)
    svc = build_service(setup)
    with sqlite3.connect(jobs / "jobs.sqlite3") as db:
        db.execute("""
            CREATE TRIGGER reject_queued BEFORE INSERT ON tasks
            BEGIN SELECT RAISE(FAIL, 'simulated insert rejection'); END
        """)
    with pytest.raises(JobIntakeError) as error:
        svc.submit(InputSource.upload(source))
    assert error.value.code == "QUEUE_PERSIST_FAILED"
    assert db_rows(jobs) == []
    assert list(jobs.glob("*/input/*")) == []

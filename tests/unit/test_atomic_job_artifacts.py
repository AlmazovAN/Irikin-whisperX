"""PR18 / P17: atomic complete result sets and indefinite safe retention.

Filesystem, SQLite and threads are real; only FFprobe/ASR are fake.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import json
import sqlite3
from pathlib import Path
from threading import Event
import time
import wave

import pytest

from modules.jobs.artifacts import (
    ArtifactError, MANIFEST_VERSION, publish_result_set, verify_result_set,
)
from modules.jobs.inputs import InputSource
from modules.jobs.service import JobIntakeError, JobService
from modules.jobs.worker import JobWorker

pytestmark = pytest.mark.fast


@dataclass
class Word:
    word: str = "Здравствуйте"
    start: float = 0.0
    end: float = 0.3
    speaker: str = "SPEAKER_00"


@dataclass
class Segment:
    start: float = 0.0
    end: float = 0.3
    text: str = "Здравствуйте"
    speaker: str = "SPEAKER_00"
    words: tuple = (Word(),)


@dataclass
class Result:
    input_name: str = "совещание.wav"
    segments: tuple = (Segment(),)
    elapsed_seconds: float = 0.5
    model_size: str = "small"
    language: str = "ru"
    diarization_requested: bool = True


@pytest.fixture(autouse=True)
def stub_ffprobe(monkeypatch):
    monkeypatch.setattr(
        "modules.jobs.inputs._probe_audio",
        lambda path, *, binary, timeout: (0.4, "pcm_s16le"),
    )


@pytest.fixture
def job(tmp_path):
    upload = tmp_path / "approved"
    upload.mkdir()
    root = tmp_path / "data" / "jobs"
    svc = JobService(storage_root=root, local_roots=(upload,))
    def create(name="совещание"):
        audio = upload / (name + ".wav")
        with wave.open(str(audio), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\x00\x00" * 6400)
        return svc.submit(InputSource.local_file(audio))
    return svc, root, create


def wait_for(svc, identifier, status):
    end = time.monotonic() + 6
    while time.monotonic() < end:
        row = svc.get(identifier)
        if row and row.status == status:
            return row
        time.sleep(0.01)
    pytest.fail(f"job did not reach {status}")


def test_atomic_publish_complete_pair_and_manifest_checksum(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    published = publish_result_set(Result(), job_dir, attempt=1)
    folder = job_dir / "result" / "attempt-1"
    assert published.files.json_path == folder / "transcript.json"
    assert published.files.txt_path == folder / "transcript.txt"
    assert published.manifest_path == folder / "manifest.json"
    manifest = json.loads(published.manifest_path.read_text("utf-8"))
    assert manifest["version"] == MANIFEST_VERSION
    assert manifest["attempt"] == 1
    assert set(manifest["files"]) == {"transcript.json", "transcript.txt"}
    assert manifest["files"]["transcript.json"]["bytes"] > 0
    assert verify_result_set(folder, attempt=1) == published
    assert not list((job_dir / "result").glob(".*pending*"))
    assert "SPEAKER_00" in published.files.txt_path.read_text("utf-8")


def test_failed_second_file_write_cannot_publish_first_file(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    import modules.jobs.artifacts as m
    original = m._write_fsynced
    def fail_second(path, data):
        if path.name == "transcript.txt":
            raise OSError("simulated disk full")
        return original(path, data)
    monkeypatch.setattr(m, "_write_fsynced", fail_second)
    with pytest.raises(ArtifactError) as error:
        publish_result_set(Result(), job_dir, attempt=1)
    assert error.value.code == "RESULT_WRITE_FAILED"
    assert not (job_dir / "result" / "attempt-1").exists()
    assert list((job_dir / "result").iterdir()) == []


def test_failed_directory_rename_leaves_no_published_pair(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    original = Path.rename
    def reject_rename(path, target):
        if path.name.startswith(".attempt-"):
            raise OSError("disk failure on rename")
        return original(path, target)
    monkeypatch.setattr(Path, "rename", reject_rename)
    with pytest.raises(ArtifactError) as error:
        publish_result_set(Result(), job_dir, attempt=1)
    assert error.value.code == "RESULT_WRITE_FAILED"
    assert list((job_dir / "result").iterdir()) == []


def test_failed_formatting_creates_no_result_file(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    with pytest.raises(ValueError):
        publish_result_set(Result(segments=()), job_dir, attempt=1)
    assert list((job_dir / "result").iterdir()) == []


def test_duplicate_attempt_never_overwrites_existing_data(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    one = publish_result_set(Result(), job_dir, attempt=1)
    before = one.files.json_path.read_bytes()
    with pytest.raises(ArtifactError) as error:
        publish_result_set(Result(), job_dir, attempt=1)
    assert error.value.code == "RESULT_EXISTS"
    assert one.files.json_path.read_bytes() == before
    two = publish_result_set(Result(), job_dir, attempt=2)
    assert two.files.json_path != one.files.json_path
    assert verify_result_set(one.files.json_path.parent, attempt=1) == one
    assert verify_result_set(two.files.json_path.parent, attempt=2) == two


@pytest.mark.parametrize("name", ["transcript.json", "transcript.txt", "manifest.json"])
def test_corruption_or_missing_file_is_detected(tmp_path, name):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    pub = publish_result_set(Result(), job_dir, attempt=1)
    file = pub.manifest_path.parent / name
    if name == "manifest.json":
        file.write_text('{"version":1,"attempt":1}', encoding="utf-8")
    else:
        file.write_bytes(b"tampered bytes")
    with pytest.raises(ArtifactError) as error:
        verify_result_set(pub.manifest_path.parent, attempt=1)
    assert error.value.code == "RESULT_CORRUPT"


def test_missing_manifest_is_never_accepted(tmp_path):
    folder = tmp_path / "job"
    folder.mkdir()
    bundle = publish_result_set(Result(), folder, attempt=1)
    bundle.manifest_path.unlink()
    with pytest.raises(ArtifactError, match="integrity"):
        verify_result_set(bundle.files.json_path.parent, attempt=1)


def test_symlink_output_root_is_rejected_without_writes(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (job_dir / "result").symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("Symlinks are not supported on this test system")
    with pytest.raises(ArtifactError) as error:
        publish_result_set(Result(), job_dir, attempt=1)
    assert error.value.code == "UNSAFE_RESULT_DIR"
    assert list(outside.iterdir()) == []


def test_verified_access_requires_completed_and_uses_trusted_db_path(job):
    svc, root, add = job
    ticket = add()
    with pytest.raises(JobIntakeError) as not_ready:
        svc.get_artifact(ticket.identifier, format="txt")
    assert not_ready.value.code == "ARTIFACT_NOT_READY"
    with pytest.raises(JobIntakeError):
        svc.get_artifact(ticket.identifier, format="pdf")
    with pytest.raises(JobIntakeError):
        svc.get_artifact("../secret", format="json")
    class Fake:
        def transcribe(self, source):
            return Result(input_name=source.safe_filename)
    worker = JobWorker(svc, transcriber_factory=Fake, poll_interval=0.01)
    worker.start()
    try:
        wait_for(svc, ticket.identifier, "COMPLETED")
    finally:
        assert worker.stop(timeout=5)
    txt = svc.get_artifact(ticket.identifier, format="txt")
    js = svc.get_artifact(ticket.identifier, format="json")
    assert txt.parent == js.parent
    assert txt.parent.name == "attempt-1"
    assert json.loads(js.read_text(encoding="utf-8"))["segments"][0]["text"]
    assert svc.saved_input(ticket.identifier).is_file()
    with closing(svc._connect()) as db:
        row = db.execute("SELECT output_txt_path, output_json_path FROM tasks WHERE uuid=?",
                         (ticket.identifier,)).fetchone()
    assert (root / row["output_txt_path"]) == txt
    assert (root / row["output_json_path"]) == js
    assert not hasattr(svc.get(ticket.identifier), "output_txt_path")

    # A new service instance retrieves the same persistent immutable data.
    new = JobService(storage_root=root)
    assert new.get_artifact(ticket.identifier, format="txt").read_bytes() == txt.read_bytes()
    assert new.get(ticket.identifier).status == "COMPLETED"
    assert not list(root.glob("*/result/.*pending*"))


def test_modified_completed_transcript_is_refused_on_new_request(job):
    svc, _, add = job
    ticket = add()
    class Fake:
        def transcribe(self, source):
            return Result(input_name=source.safe_filename)
    worker = JobWorker(svc, transcriber_factory=Fake, poll_interval=0.01)
    worker.start()
    try:
        wait_for(svc, ticket.identifier, "COMPLETED")
    finally:
        assert worker.stop(timeout=5)
    saved = svc.get_artifact(ticket.identifier, format="txt")
    saved.write_text("modified without manifest", encoding="utf-8")
    with pytest.raises(ArtifactError) as error:
        svc.get_artifact(ticket.identifier, format="txt")
    assert error.value.code == "RESULT_CORRUPT"


def test_invalid_database_result_locator_cannot_escape_root(job):
    svc, _, add = job
    ticket = add()
    class Fake:
        def transcribe(self, source):
            return Result(input_name=source.safe_filename)
    worker = JobWorker(svc, transcriber_factory=Fake, poll_interval=0.01)
    worker.start()
    try:
        wait_for(svc, ticket.identifier, "COMPLETED")
    finally:
        assert worker.stop(timeout=5)
    with closing(svc._connect()) as conn:
        with conn:
            conn.execute(
                "UPDATE tasks SET output_txt_path=? WHERE uuid=?",
                ("../../external-data.txt", ticket.identifier),
            )
    with pytest.raises(ArtifactError) as error:
        svc.get_artifact(ticket.identifier, format="txt")
    assert error.value.code == "RESULT_CORRUPT"


def test_no_automatic_retention_deletion_on_restart(job):
    svc, root, add = job
    ticket = add()
    class Fake:
        def transcribe(self, source):
            return Result(input_name=source.safe_filename)
    worker = JobWorker(svc, transcriber_factory=Fake, poll_interval=0.01)
    worker.start()
    try:
        wait_for(svc, ticket.identifier, "COMPLETED")
    finally:
        assert worker.stop(timeout=5)
    saved_txt = svc.get_artifact(ticket.identifier, format="txt")
    saved_input = svc.saved_input(ticket.identifier)
    JobService(storage_root=root)  # Simulate a fresh process opening storage.
    assert saved_txt.is_file()
    assert saved_input.is_file()
    assert svc.get(ticket.identifier).status == "COMPLETED"

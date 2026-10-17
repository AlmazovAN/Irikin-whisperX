"""P23: verified TXT availability and fallback download.

Exercises real SQLite, atomic artifact files and worker lifecycle with fake ASR.
No browser, Gradio imports, models, network access or private recordings.
"""
from __future__ import annotations

from dataclasses import dataclass
import sqlite3
import time
import wave
from pathlib import Path
from uuid import uuid4

import pytest

from modules.jobs.inputs import InputSource
from modules.jobs.service import JobService
from modules.jobs.worker import JobWorker
from modules.ui.simple_mode import SimpleModeController, _AUTO_DOWNLOAD_JS

pytestmark = pytest.mark.fast


@dataclass
class Segment:
    start: float = 12.0
    end: float = 13.0
    text: str = "SPEAKER_00|Добрый день"
    speaker: str = "SPEAKER_00"


@dataclass
class FakeResult:
    segments: tuple = (Segment(),)
    input_name: str = "пример.wav"
    elapsed_seconds: float = 1.0
    model_size: str = "small"
    language: str = "ru"
    diarization_requested: bool = True


class FakeASR:
    def transcribe(self, source):
        return FakeResult(input_name=source.safe_filename)


@pytest.fixture(autouse=True)
def probe(monkeypatch):
    monkeypatch.setattr(
        "modules.jobs.inputs._probe_audio",
        lambda path, *, binary, timeout: (1.0, "pcm_s16le"),
    )


@pytest.fixture
def files(tmp_path, monkeypatch):
    upload = tmp_path / "uploads"
    upload.mkdir()
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(upload))
    audio = upload / "sample.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 1600)
    root = tmp_path / "data" / "jobs"
    jobs = JobService(storage_root=root, upload_roots=(upload,))
    worker = JobWorker(jobs, transcriber_factory=FakeASR, poll_interval=0.01)
    controller = SimpleModeController(jobs, worker)
    return jobs, worker, controller, audio


def run_to_completion(jobs, worker, identifier):
    worker.start()
    try:
        limit = time.monotonic() + 5
        while time.monotonic() < limit:
            state = jobs.get_status(identifier)
            if state.status in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.01)
        assert state.status == "COMPLETED", state.error_code
    finally:
        assert worker.stop(timeout=5)


def test_download_appears_only_after_verified_commit(files):
    jobs, worker, controller, audio = files
    job = jobs.submit(InputSource.upload(audio))
    assert controller.poll_with_download(job.identifier)[-1] is None
    run_to_completion(jobs, worker, job.identifier)

    id_out, status, stage, progress, file = controller.poll_with_download(job.identifier)
    assert id_out == job.identifier
    assert "Готово" in status and stage == "Готово" and progress == 100
    assert file == str(jobs.get_artifact(job.identifier, format="txt"))
    assert Path(file).read_text(encoding="utf-8") == (
        "[00:00:12] [SPEAKER_00]: Добрый день\n"
    )
    assert jobs.saved_input(job.identifier).is_file()
    assert jobs.get_artifact(job.identifier, format="json").is_file()

    # Another controller session/reload can reopen the same verified file.
    another_jobs = JobService(storage_root=jobs.storage.root)
    another_worker = JobWorker(another_jobs, transcriber_factory=FakeASR)
    fresh = SimpleModeController(another_jobs, another_worker)
    assert fresh.poll_with_download(job.identifier)[-1] == file


def test_failed_corrupt_or_unrecognized_job_never_exposes_file(files):
    jobs, worker, controller, audio = files
    assert controller.poll_with_download("../../etc/passwd")[-1] is None
    assert controller.poll_with_download(str(uuid4()))[-1] is None
    ticket = jobs.submit(InputSource.upload(audio))
    with jobs._connect() as conn:
        conn.execute(
            "UPDATE tasks SET status='FAILED', stage='FAILED' WHERE uuid=?",
            (ticket.identifier,),
        )
        conn.commit()
    assert controller.poll_with_download(ticket.identifier)[-1] is None


def test_tampered_txt_disables_download_without_false_link(files):
    jobs, worker, controller, audio = files
    ticket = jobs.submit(InputSource.upload(audio))
    run_to_completion(jobs, worker, ticket.identifier)
    path = jobs.get_artifact(ticket.identifier, format="txt")
    path.write_text("Изменено", encoding="utf-8")
    _, status, stage, progress, download = controller.poll_with_download(ticket.identifier)
    assert download is None
    assert "повреждён" in status
    assert stage == "Готово" and progress == 100
    assert jobs.saved_input(ticket.identifier).exists()


def test_manual_download_survives_failed_new_upload(files):
    jobs, worker, controller, audio = files
    ticket = jobs.submit(InputSource.upload(audio))
    run_to_completion(jobs, worker, ticket.identifier)
    uuid, msg, stage, progress, saved, path = controller.submit_and_download(
        None, ticket.identifier,
    )
    assert uuid == saved == ticket.identifier
    assert "Выберите аудиофайл" in msg
    assert stage == "Готово" and progress == 100
    assert path == str(jobs.get_artifact(ticket.identifier, format="txt"))


def test_new_submission_clears_old_download(files):
    jobs, _, controller, audio = files
    ticket = jobs.submit(InputSource.upload(audio))
    # Fresh submission needs a live worker owner. This stub only posts jobs.
    class RunningWorker:
        def __init__(self):
            self.jobs = jobs
            self.is_running = True
            self.wakes = 0
        def wake(self):
            self.wakes += 1
    owner = RunningWorker()
    ctl = SimpleModeController(jobs, owner)
    accepted, message, stage, progress, stored, file = ctl.submit_and_download(
        str(audio), ticket.identifier,
    )
    assert accepted != ticket.identifier and stored == accepted
    assert file is None and stage == "В очереди" and progress == 0
    assert owner.wakes == 1


def test_only_gradio_file_endpoint_is_needed_not_whole_directory_allowlist():
    import inspect
    from modules.ui import simple_mode
    code = inspect.getsource(simple_mode.launch_simple)
    assert "allowed_paths=" not in code
    assert "share=False" in code
    assert "server_name=host" in code
    assert "api_open=False" in code
    assert "download.change(" in inspect.getsource(simple_mode.build_blocks)
    assert "localStorage" in _AUTO_DOWNLOAD_JS
    assert ".click()" in _AUTO_DOWNLOAD_JS
    assert "mvp-download-txt" in _AUTO_DOWNLOAD_JS
    assert "window.localStorage.setItem" in _AUTO_DOWNLOAD_JS

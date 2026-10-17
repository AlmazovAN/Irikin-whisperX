"""P21: UUID reconnect and honest status snapshots, with no real ASR or browser.

The SQLite database and stored WAV bytes are real. Fake probe/worker only.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3
import wave
from uuid import uuid4

import pytest

from modules.jobs.service import JobService
from modules.ui.simple_mode import SimpleModeController, upload_directory

pytestmark = pytest.mark.fast


class FakeWorker:
    def __init__(self, jobs: JobService):
        self.jobs = jobs
        self.is_running = True
        self.wakes = 0

    def wake(self):
        self.wakes += 1


@pytest.fixture
def local(tmp_path, monkeypatch):
    stage = tmp_path / "uploads"
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(stage))
    monkeypatch.setattr(
        "modules.jobs.inputs._probe_audio",
        lambda path, *, binary, timeout: (2.0, "pcm_s16le"),
    )
    root = tmp_path / "jobs"
    jobs = JobService(storage_root=root, upload_roots=(upload_directory(),))
    worker = FakeWorker(jobs)
    controller = SimpleModeController(jobs, worker)
    path = stage / "hash" / "meeting.wav"
    path.parent.mkdir()
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\0\0" * 2000)
    return controller, jobs, worker, path, root


def set_status(jobs, job_id, *, status, stage, progress, error_code=None):
    with closing(sqlite3.connect(str(jobs.storage.database_path))) as conn:
        with conn:
            conn.execute(
                "UPDATE tasks SET status=?, stage=?, progress=?, error_code=? WHERE uuid=?",
                (status, stage, progress, error_code, job_id),
            )


def test_submit_sets_browser_uuid_and_page_reload_reads_sqlite(local):
    controller, jobs, worker, path, root = local
    identifier, message, stage, progress, stored_id = controller.submit_and_store(str(path), "")
    assert len(identifier) == 36 and stored_id == identifier
    assert "QUEUED" in message and stage == "В очереди" and progress == 0
    assert worker.wakes == 1

    # A fresh UI/controller after browser refresh gets a new SQLite connection.
    fresh_jobs = JobService(storage_root=root)
    refreshed = SimpleModeController(fresh_jobs, FakeWorker(fresh_jobs))
    assert refreshed.poll(stored_id) == (
        identifier, "Задание ожидает обработки.", "В очереди", 0,
    )

    # Worker changes state independently of any Gradio session.
    set_status(jobs, identifier, status="IN_PROGRESS", stage="ASR", progress=0.265)
    assert refreshed.poll(stored_id) == (
        identifier, "Обработка выполняется на этом компьютере.", "Распознавание", 26,
    )
    assert worker.wakes == 1  # Polling does not enqueue/retry inference.

    set_status(jobs, identifier, status="IN_PROGRESS", stage="DIARIZE", progress=0.85)
    assert refreshed.poll(stored_id)[2:] == ("Определение спикеров", 85)

    set_status(jobs, identifier, status="COMPLETED", stage="DONE", progress=1.0)
    assert refreshed.poll(stored_id) == (
        identifier, "Готово. Результат сохранён на ПК.", "Готово", 100,
    )


def test_cancel_pending_failure_and_interrupted_are_not_false_success(local):
    controller, jobs, _, path, _ = local
    identifier, *_ = controller.submit_and_store(str(path), "")
    set_status(jobs, identifier, status="IN_PROGRESS", stage="ASR", progress=0.12,
               error_code="CANCEL_REQUESTED")
    uid, message, stage, progress = controller.poll(identifier)
    assert uid == identifier and "Отмена запрошена" in message
    assert stage == "Распознавание" and progress == 12

    set_status(jobs, identifier, status="FAILED", stage="FAILED", progress=0.85,
               error_code="OUT_OF_MEMORY")
    _, message, stage, progress = controller.poll(identifier)
    assert "OUT_OF_MEMORY" in message and "Готово" not in message
    assert stage == "Ошибка" and progress == 85

    set_status(jobs, identifier, status="INTERRUPTED", stage="INTERRUPTED", progress=0.85)
    _, message, stage, progress = controller.poll(identifier)
    assert "прервана" in message and stage == "Прервано" and progress == 85


def test_invalid_or_missing_browser_id_is_never_echoed(local):
    controller, jobs, worker, _, _ = local
    assert controller.poll("") == ("", "Выберите аудиофайл для обработки.", "—", 0)
    visible, message, stage, progress = controller.poll("../../secret")
    assert visible == "" and "Некорректный" in message
    assert "../../secret" not in message
    visible, message, _, _ = controller.poll(str(uuid4()))
    assert visible == "" and "не найдено" in message
    assert worker.wakes == 0
    assert jobs.list_jobs() == ()


def test_rejected_upload_does_not_erase_previous_browser_job(local):
    controller, jobs, worker, path, _ = local
    existing, *_ = controller.submit_and_store(str(path), "")
    before = len(jobs.list_jobs())
    visible, error, stage, milestone, stored = controller.submit_and_store(None, existing)
    assert visible == stored == existing
    assert "Выберите аудиофайл" in error
    assert stage == "В очереди" and milestone == 0
    assert len(jobs.list_jobs()) == before and worker.wakes == 1


def test_sqlite_failure_does_not_expose_exception(monkeypatch, local):
    controller, jobs, worker, path, _ = local
    identifier, *_ = controller.submit_and_store(str(path), "")

    def fail(_):
        raise sqlite3.OperationalError("SECRET PRIVATE PATH")

    monkeypatch.setattr(jobs, "get_status", fail)
    visible, message, stage, progress = controller.poll(identifier)
    assert visible == identifier and "временно недоступен" in message
    assert "SECRET" not in message and stage == "—" and progress == 0
    assert worker.wakes == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -2, 8])
def test_milestones_are_clamped_and_never_claim_completion(local, value):
    controller, jobs, _, path, _ = local
    identifier, *_ = controller.submit_and_store(str(path), "")
    set_status(jobs, identifier, status="IN_PROGRESS", stage="ASR", progress=value)
    _, msg, stage, percent = controller.poll(identifier)
    assert 0 <= percent <= 100
    assert "выполняется" in msg and stage == "Распознавание"

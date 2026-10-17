"""P20/P21: durable Gradio intake, SQLite polling and browser reconnect.

Gradio is imported only when constructing the interface. All heavy inference
is owned exclusively by the single background JobWorker, never a UI callback.
Downloads and history belong to following PRs.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import sqlite3
from uuid import UUID
from typing import Any

from modules.jobs.inputs import AUDIO_EXTENSIONS, InputSource, InputValidationError
from modules.jobs.service import JobIntakeError, JobService
from modules.jobs.storage import DEFAULT_JOB_ROOT, JobStorageError
from modules.jobs.worker import JobWorker

_ERROR_TYPES = (InputValidationError, JobIntakeError, JobStorageError)
_SUPPORTED_FILE_TYPES = [".mp3", ".wav", ".m4a", ".ogg"]
# Only an opaque job UUID is stored in browser localStorage, never audio, paths
# or transcripts. Stable keys let BrowserState survive localhost restarts.
_BROWSER_STORAGE_KEY = "irikin-whisperx.simple.last-job.v1"
_BROWSER_STORAGE_SECRET = "irikin-whisperx.local-uuid-only.v1"
_STAGE_LABELS = {
    "QUEUED": "В очереди",
    "PREPARING": "Подготовка",
    "VAD": "Выделение речи",
    "ASR": "Распознавание",
    "ALIGN": "Выравнивание таймкодов",
    "DIARIZE": "Определение спикеров",
    "EXPORT": "Сохранение результата",
    "DONE": "Готово",
    "FAILED": "Ошибка",
    "INTERRUPTED": "Прервано",
    "CANCELLED": "Отменено",
}
_STATUS_LABELS = {
    "QUEUED": "Задание ожидает обработки.",
    "IN_PROGRESS": "Обработка выполняется на этом компьютере.",
    "COMPLETED": "Готово. Результат сохранён на ПК.",
    "FAILED": "Ошибка обработки. Исходный файл сохранён.",
    "INTERRUPTED": "Обработка прервана при остановке сервера. Исходный файл сохранён.",
    "CANCELLED": "Задание отменено.",
}


def upload_directory() -> Path:
    """Use the same trusted temporary directory configured for Gradio uploads.

    This path comes from local server configuration, NOT from browser input.
    Reject symlinked temp directories and filesystem roots.
    """
    configured = os.environ.get("GRADIO_TEMP_DIR")
    directory = Path(configured) if configured else (
        Path(__file__).resolve().parents[2] / "data" / "ui_uploads"
    )
    directory = directory.expanduser()
    if directory.is_symlink():
        raise JobIntakeError("UNSAFE_UPLOAD_ROOT", "Upload directory must not be a symlink")
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory = directory.resolve(strict=True)
    except (OSError, RuntimeError):
        raise JobIntakeError("UPLOAD_ROOT_FAILED", "Cannot initialize upload storage") from None
    if directory == Path(directory.anchor) or not directory.is_dir():
        raise JobIntakeError("UNSAFE_UPLOAD_ROOT", "Upload storage is unsafe")
    return directory


class SimpleModeController:
    """Submit-only UI adapter, independent of Gradio and ML imports."""

    def __init__(self, jobs: JobService, worker: JobWorker):
        if worker.jobs is not jobs:
            raise ValueError("Simple Mode must use exactly the worker's JobService")
        self.jobs = jobs
        self.worker = worker

    @staticmethod
    def _source_path(upload: Any) -> str:
        """Accept only a Gradio upload's server-side filepath/name."""
        if isinstance(upload, (str, Path)):
            return os.fspath(upload)
        # Gradio may provide a NamedString-style file object.
        name = getattr(upload, "name", None)
        if isinstance(name, str):
            return name
        raise InputValidationError("FILE_REQUIRED", "Choose an audio file first")

    def submit(self, upload: Any) -> tuple[str, str]:
        """Synchronous durable intake only; no ASR work in the UI thread."""
        if upload is None or upload == "":
            return "", "Выберите аудиофайл."
        if not self.worker.is_running:
            return "", "Ошибка WORKER_UNAVAILABLE: обработчик не запущен."
        try:
            path = self._source_path(upload)
            ticket = self.jobs.submit(InputSource.upload(path))
        except _ERROR_TYPES as exc:
            # Only stable codes, never exception text, system paths or audio.
            return "", f"Ошибка {exc.code}: файл не принят."
        except (OSError, ValueError, TypeError):
            return "", "Ошибка SUBMIT_FAILED: файл не принят."
        self.worker.wake()
        return ticket.identifier, "В очереди (QUEUED). Файл сохранён на этом компьютере."

    def poll(self, identifier: str) -> tuple[str, str, str, int]:
        """Read fresh SQLite state by canonical UUID; no inference or file serving.

        Progress represents persisted milestones, not elapsed time or ETA.
        Never echo arbitrary browser input, paths, model text or exception text.
        """
        if not identifier:
            return "", "Выберите аудиофайл для обработки.", "—", 0
        if not isinstance(identifier, str):
            return "", "Некорректный идентификатор задания.", "—", 0
        try:
            if str(UUID(identifier)) != identifier:
                raise ValueError("Noncanonical identifier")
        except (ValueError, TypeError, AttributeError):
            return "", "Некорректный идентификатор задания.", "—", 0
        try:
            ticket = self.jobs.get_status(identifier)
        except (sqlite3.Error, OSError, JobIntakeError, JobStorageError):
            return identifier, "Статус временно недоступен. Повторим запрос.", "—", 0
        if ticket is None:
            return "", "Задание не найдено в локальной базе.", "—", 0

        stage = _STAGE_LABELS.get(ticket.stage, "Неизвестный этап")
        status = _STATUS_LABELS.get(ticket.status, "Состояние неизвестно.")
        if ticket.status == "IN_PROGRESS" and ticket.error_code == "CANCEL_REQUESTED":
            status = "Отмена запрошена. Ожидается безопасная точка остановки."
        elif ticket.status == "FAILED" and isinstance(ticket.error_code, str):
            # Only stable internal codes, never arbitrary exception strings.
            if re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", ticket.error_code):
                status += f" Код: {ticket.error_code}."
        try:
            value = float(ticket.progress)
            milestone = round(max(0.0, min(1.0, value)) * 100) if math.isfinite(value) else 0
        except (TypeError, ValueError):
            milestone = 0
        if ticket.status == "COMPLETED":
            milestone = 100
        return ticket.identifier, status, stage, milestone

    def submit_and_store(
        self, upload: Any, previous_identifier: str,
    ) -> tuple[str, str, str, int, str]:
        """Queue exactly once and store its UUID in browser-local state.

        A rejected new upload must not erase the previous accepted job.
        """
        identifier, message = self.submit(upload)
        if identifier:
            return identifier, message, _STAGE_LABELS["QUEUED"], 0, identifier
        previous = self.poll(previous_identifier)
        retained = previous[0]
        return retained, message, previous[2], previous[3], retained


def create_runtime(*, job_root: str | Path = DEFAULT_JOB_ROOT) -> tuple[SimpleModeController, JobWorker]:
    """Share exactly one durable service and one single-owner worker."""
    uploads = upload_directory()
    service = JobService(storage_root=job_root, upload_roots=(uploads,))
    worker = JobWorker(service)
    return SimpleModeController(service, worker), worker


def build_blocks(controller: SimpleModeController, *, gradio: Any = None) -> Any:
    """Build the minimal interface, compatible with common Gradio Blocks APIs."""
    if gradio is None:
        import gradio as gradio
    if not isinstance(controller, SimpleModeController):
        raise TypeError("Expected SimpleModeController")

    with gradio.Blocks() as blocks:
        gradio.Markdown(
            "# WhisperX — локальная расшифровка\n"
            "Перетащите аудиофайл, нажмите **Начать обработку**. "
            "Распознавание выполняется **на этом компьютере**."
        )
        with gradio.Tabs():
            with gradio.Tab("Simple"):
                gradio.Markdown(
                    "**Модель:** Whisper Small · **Язык:** русский (ru) · "
                    "**Режим:** transcribe · **Спикеры:** включены"
                )
                uploaded = gradio.File(
                    label="Аудиофайл (MP3, WAV, M4A, OGG)",
                    file_types=_SUPPORTED_FILE_TYPES,
                    type="filepath",
                    file_count="single",
                )
                # Same localhost origin + stable key/secret lets UUID survive
                # refresh, tab close and server restart. Not an auth token.
                browser_job = gradio.BrowserState(
                    "", storage_key=_BROWSER_STORAGE_KEY,
                    secret=_BROWSER_STORAGE_SECRET,
                )
                submit = gradio.Button("Начать обработку", variant="primary")
                status = gradio.Textbox(label="Состояние", interactive=False)
                identifier = gradio.Textbox(label="ID задания", interactive=False)
                stage = gradio.Textbox(label="Этап обработки", interactive=False)
                progress = gradio.Slider(
                    minimum=0, maximum=100, step=1, value=0,
                    label="Контрольные точки конвейера (%)",
                    interactive=False,
                )
                gradio.Markdown(
                    "Шкала отображает пройденные этапы, **не точное время или "
                    "процент оставшейся вычислительной работы**. "
                    "На ASR она может долго не изменяться."
                )
                submit.click(
                    fn=controller.submit_and_store,
                    inputs=[uploaded, browser_job],
                    outputs=[identifier, status, stage, progress, browser_job],
                    api_name=False,
                )
                # Load and tick obtain *fresh* snapshots via SQLite. The timer
                # belongs to the browser view, never to the background worker.
                blocks.load(
                    fn=controller.poll,
                    inputs=[browser_job],
                    outputs=[identifier, status, stage, progress],
                    queue=False, show_progress="hidden", api_name=False,
                )
                refresh = gradio.Timer(value=2.0, active=True)
                refresh.tick(
                    fn=controller.poll,
                    inputs=[browser_job],
                    outputs=[identifier, status, stage, progress],
                    queue=False, show_progress="hidden", api_name=False,
                )
                gradio.Markdown(
                    "Статус восстанавливается по ID после обновления или "
                    "повторного открытия вкладки. Обработка продолжится, "
                    "пока работает локальный сервер. Скачивание TXT — P23."
                )
            with gradio.Tab("Advanced"):
                gradio.Markdown(
                    "Старый интерфейс с расширенными настройками сохранён. "
                    "Чтобы открыть его, **остановите текущий сервер** "
                    "и запустите `Start.bat --ui-mode advanced`. "
                    "В одном запуске используется только один режим, "
                    "чтобы не создавать две конкурирующие модели."
                )
    return blocks


def launch_simple(args: Any, *, gradio: Any = None) -> None:
    """Start one worker, then one local Gradio server; stop cleanly on exit."""
    from modules.utils.local_security import validate_launch_settings
    host = validate_launch_settings(
        server_name=args.server_name, share=args.share, api_open=args.api_open,
    )
    controller, worker = create_runtime()
    interface = build_blocks(controller, gradio=gradio)
    worker.start()  # Must own the lock BEFORE browser can submit a job.
    try:
        interface.queue(api_open=False).launch(
            share=False, server_name=host, server_port=args.server_port,
            auth=(args.username, args.password)
            if args.username and args.password else None,
            root_path=args.root_path, inbrowser=args.inbrowser,
            ssl_verify=args.ssl_verify,
            ssl_keyfile=args.ssl_keyfile,
            ssl_keyfile_password=args.ssl_keyfile_password,
            ssl_certfile=args.ssl_certfile,
        )
    finally:
        # Do not forcibly kill running inference on browser reload; only
        # server shutdown calls this. Crash recovery is handled by PR19.
        worker.stop(wait=True)

"""PR21 / P20: one-screen job intake with no direct WhisperX invocation.

Gradio is imported only when constructing the interface. All heavy inference
is owned exclusively by the single background JobWorker, never a UI callback.
Job polling/reconnect, downloads and history belong to following PRs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from modules.jobs.inputs import AUDIO_EXTENSIONS, InputSource, InputValidationError
from modules.jobs.service import JobIntakeError, JobService
from modules.jobs.storage import DEFAULT_JOB_ROOT, JobStorageError
from modules.jobs.worker import JobWorker

_ERROR_TYPES = (InputValidationError, JobIntakeError, JobStorageError)
_SUPPORTED_FILE_TYPES = [".mp3", ".wav", ".m4a", ".ogg"]


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
                submit = gradio.Button("Начать обработку", variant="primary")
                status = gradio.Textbox(label="Состояние", interactive=False)
                identifier = gradio.Textbox(label="ID задания", interactive=False)
                submit.click(
                    fn=controller.submit,
                    inputs=[uploaded],
                    outputs=[identifier, status],
                    api_name=False,
                )
                gradio.Markdown(
                    "Обработка продолжается после закрытия вкладки, пока работает "
                    "сервер. Восстановление статуса и скачивание TXT появятся "
                    "в следующих обновлениях интерфейса."
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

"""P20-P24: durable intake, SQLite history and verified transcript previews.

Gradio is imported only when constructing the interface. All heavy inference
is owned exclusively by the single background JobWorker, never a UI callback.
All heavy inference remains in the existing single JobWorker.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import sqlite3
from uuid import UUID
from typing import Any

from modules.jobs.artifacts import ArtifactError
from modules.jobs.inputs import AUDIO_EXTENSIONS, InputSource, InputValidationError
from modules.jobs.service import JobIntakeError, JobService
from modules.jobs.storage import DEFAULT_JOB_ROOT, JobStorageError
from modules.jobs.worker import JobWorker

_ERROR_TYPES = (InputValidationError, JobIntakeError, JobStorageError)
_SUPPORTED_FILE_TYPES = [".mp3", ".wav", ".m4a", ".ogg"]
_PREVIEW_BYTES = 12 * 1024
_HISTORY_LIMIT = 30
# Only an opaque job UUID is stored in browser localStorage, never audio, paths
# or transcripts. Stable keys let BrowserState survive localhost restarts.
_BROWSER_STORAGE_KEY = "irikin-whisperx.simple.last-job.v1"
_BROWSER_STORAGE_SECRET = "irikin-whisperx.local-uuid-only.v1"
# The browser attempts automatic download once per completed UUID. This is only
# a convenience; popup/download policies may block it. Manual button remains.
_DOWNLOAD_ELEM_ID = "mvp-download-txt"
_AUTO_DOWNLOAD_JS = """
(job_id) => {
    if (typeof job_id !== "string" ||
        !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(job_id)) {
        return [];
    }
    const storageKey = "irikin-whisperx.simple.auto-txt.v1";
    try {
        if (window.localStorage.getItem(storageKey) === job_id) return [];
        const container = document.getElementById("mvp-download-txt");
        if (!container) return [];
        const link = (container.tagName === "A" && container.href)
            ? container : container.querySelector("a[href]");
        const button = container.querySelector("button:not([disabled])");
        const target = link || button;
        if (!target || target.disabled) return [];
        // Record the attempt BEFORE clicking; browser success cannot be read.
        window.localStorage.setItem(storageKey, job_id);
        target.click();
    } catch (_error) {
        // Private browsing or blocked storage/download is not a job failure.
    }
    return [];
}
"""
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
    """Local UI adapter independent of Gradio and ML imports."""

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

    def verified_txt_path(self, identifier: str) -> str | None:
        """Return only a verified path from the durable completed job contract.

        Gradio serves returned file outputs through its own localhost file
        endpoint. Never accept a browser-supplied path or expose data/jobs
        wholesale via allowed_paths.
        """
        try:
            file = self.jobs.get_artifact(identifier, format="txt")
        except (JobIntakeError, JobStorageError, ArtifactError, sqlite3.Error,
                OSError, TypeError, ValueError):
            return None
        return str(file)

    def poll_with_download(self, identifier: str) -> tuple[str, str, str, int, str | None]:
        """Extend P21 status polling with a ready-to-download TXT, if verified."""
        status = self.poll(identifier)
        canonical_id = status[0]
        if not canonical_id:
            return (*status, None)
        try:
            ticket = self.jobs.get_status(canonical_id)
        except (JobIntakeError, JobStorageError, sqlite3.Error, OSError):
            return (*status, None)
        if ticket is None or ticket.status != "COMPLETED":
            return (*status, None)
        file = self.verified_txt_path(canonical_id)
        if file is None:
            return (
                canonical_id,
                "Задание завершено, но файл TXT недоступен или повреждён. "
                "Исходник и запись задания сохранены.",
                status[2],
                status[3],
                None,
            )
        return (*status, file)


    def history_options(self, selected_id: str) -> tuple[list[tuple[str, str]], str | None]:
        """Most recent durable jobs, with labels not filesystem paths."""
        tickets = self.jobs.list_jobs(limit=_HISTORY_LIMIT)
        choices: list[tuple[str, str]] = []
        for ticket in tickets:
            safe_name = re.sub(r"[\x00-\x1f\x7f]", " ", ticket.file_name)[:90]
            label = (
                f"{ticket.created_at[:16]} · {safe_name} · "
                f"{ticket.status} · {ticket.model_name}"
            )
            choices.append((label, ticket.identifier))
        valid = {identifier for _, identifier in choices}
        return choices, selected_id if selected_id in valid else None

    def preview(self, identifier: str) -> str:
        """Bounded preview of SHA-verified TXT; no background ASR or file paths."""
        file = self.verified_txt_path(identifier)
        if file is None:
            return ""
        try:
            with Path(file).open("rb") as stream:
                content = stream.read(_PREVIEW_BYTES + 1)
        except OSError:
            return "Предпросмотр временно недоступен."
        more = len(content) > _PREVIEW_BYTES
        text = content[:_PREVIEW_BYTES].decode("utf-8", errors="replace")
        return text + ("\n… Показано начало TXT. Скачайте файл целиком." if more else "")

    def poll_with_preview(
        self, identifier: str,
    ) -> tuple[str, str, str, int, str | None, str]:
        data = self.poll_with_download(identifier)
        return (*data, self.preview(data[0]) if data[4] is not None else "")

    def select_history(
        self, identifier: str,
    ) -> tuple[str, str, str, int, str | None, str, str, bool]:
        """Selecting a past job updates BrowserState; never starts inference."""
        current = self.poll_with_preview(identifier)
        return (*current[:5], current[0], current[5], False)

    def submit_with_preview(
        self, upload: Any, previous_identifier: str,
    ) -> tuple[str, str, str, int, str, str | None, str]:
        result = self.submit_and_download(upload, previous_identifier)
        return (*result, self.preview(result[0]) if result[5] else "")

    def open_result_folder(self, identifier: str, *, opener: Any = None) -> str:
        """Open only a verified result directory on the local Windows host."""
        file = self.verified_txt_path(identifier)
        if file is None:
            return "Папка доступна только для завершённого задания с целым TXT."
        chosen = opener if opener is not None else getattr(os, "startfile", None)
        if not callable(chosen):
            return "Открытие папки поддерживается в Windows при запуске на этом ПК."
        try:
            chosen(str(Path(file).parent))
        except OSError:
            return "Не удалось открыть папку результата."
        return "Папка результата открыта в файловом менеджере."

    def delete_selected(
        self, selected: str, confirmed: bool, browser_id: str,
    ) -> tuple[str, str, bool]:
        """Never delete on selection or refresh; require explicit checkbox."""
        if confirmed is not True:
            return "Для удаления установите флажок подтверждения.", browser_id, False
        try:
            self.jobs.delete_terminal(selected, confirmed=True)
        except JobIntakeError as exc:
            if exc.code == "DELETE_CLEANUP_PENDING":
                return (
                    "Запись удалена, но очистка файлов не завершена. "
                    "Проверьте локальный каталог вручную.",
                    "" if selected == browser_id else browser_id,
                    False,
                )
            if exc.code == "JOB_ACTIVE":
                return "Активное задание удалить нельзя.", browser_id, False
            return f"Удаление отклонено: {exc.code}.", browser_id, False
        except (sqlite3.Error, OSError, JobStorageError):
            return "Ошибка удаления. Данные задания сохранены.", browser_id, False
        return (
            "Задание и его локальные файлы удалены.",
            "" if selected == browser_id else browser_id,
            False,
        )

    def submit_and_download(
        self, upload: Any, previous_identifier: str,
    ) -> tuple[str, str, str, int, str, str | None]:
        """Clear old download on accepted new job; retain on rejected upload."""
        previous = self.poll_with_download(previous_identifier)
        identifier, message, stage, percent, stored = self.submit_and_store(
            upload, previous_identifier,
        )
        old_file = previous[4] if identifier and identifier == previous[0] else None
        return identifier, message, stage, percent, stored, old_file

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

    def refresh_history(current_identifier: str) -> Any:
        choices, selected = controller.history_options(current_identifier)
        # A refreshed selection must never inherit a stale deletion approval.
        return gradio.update(choices=choices, value=selected), False

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
                download = gradio.DownloadButton(
                    "Скачать TXT", value=None, elem_id=_DOWNLOAD_ELEM_ID,
                )
                preview = gradio.Textbox(
                    label="Предпросмотр стенограммы (начало файла)",
                    interactive=False, lines=8,
                )
                upload_event = submit.click(
                    fn=controller.submit_with_preview,
                    inputs=[uploaded, browser_job],
                    outputs=[identifier, status, stage, progress, browser_job, download, preview],
                    api_name=False,
                )
                # Load and tick obtain *fresh* snapshots via SQLite. The timer
                # belongs to the browser view, never to the background worker.
                blocks.load(
                    fn=controller.poll_with_preview,
                    inputs=[browser_job],
                    outputs=[identifier, status, stage, progress, download, preview],
                    queue=False, show_progress="hidden", api_name=False,
                )
                refresh = gradio.Timer(value=2.0, active=True)
                refresh.tick(
                    fn=controller.poll_with_preview,
                    inputs=[browser_job],
                    outputs=[identifier, status, stage, progress, download, preview],
                    queue=False, show_progress="hidden", api_name=False,
                )
                # A changed DownloadButton is already populated with a Gradio
                # file URL when this frontend-only listener is dispatched.
                # JS is best-effort; manual DownloadButton always remains.
                download.change(
                    fn=None, inputs=[browser_job], outputs=[],
                    js=_AUTO_DOWNLOAD_JS, queue=False, api_name=False,
                )
                gradio.Markdown(
                    "После завершения браузер попробует скачать TXT один раз. "
                    "Если браузер запретит загрузку, нажмите **Скачать TXT**. "
                    "Результат уже сохранён на ПК и доступен после "
                    "повторного открытия вкладки."
                )
                gradio.Markdown(
                    "### История последних заданий\n"
                    "Список хранится в локальной SQLite. Выбор не запускает "
                    "повторную обработку."
                )
                history = gradio.Dropdown(
                    choices=[], value=None, label="Последние 30 заданий",
                    interactive=True,
                )
                refresh_button = gradio.Button("Обновить историю")
                open_button = gradio.Button("Открыть папку результата")
                confirm_delete = gradio.Checkbox(
                    label="Подтверждаю безвозвратное удаление выбранного "
                          "задания, исходного аудио и результатов",
                    value=False,
                )
                delete_button = gradio.Button("Удалить выбранное задание")
                history_message = gradio.Textbox(
                    label="Действие с заданием", interactive=False,
                )
                blocks.load(
                    fn=refresh_history, inputs=[browser_job], outputs=[history, confirm_delete],
                    queue=False, show_progress="hidden", api_name=False,
                )
                upload_event.then(
                    fn=refresh_history, inputs=[browser_job], outputs=[history, confirm_delete],
                    queue=False, show_progress="hidden", api_name=False,
                )
                refresh_button.click(
                    fn=refresh_history, inputs=[browser_job], outputs=[history, confirm_delete],
                    queue=False, api_name=False,
                )
                history.select(
                    fn=controller.select_history, inputs=[history],
                    outputs=[identifier, status, stage, progress, download,
                             browser_job, preview, confirm_delete],
                    queue=False, api_name=False,
                )
                open_button.click(
                    fn=controller.open_result_folder, inputs=[history],
                    outputs=[history_message], queue=False, api_name=False,
                )
                deleted = delete_button.click(
                    fn=controller.delete_selected,
                    inputs=[history, confirm_delete, browser_job],
                    outputs=[history_message, browser_job, confirm_delete],
                    queue=False, api_name=False,
                )
                deleted.then(
                    fn=refresh_history, inputs=[browser_job], outputs=[history, confirm_delete],
                    queue=False, show_progress="hidden", api_name=False,
                ).then(
                    fn=controller.poll_with_preview, inputs=[browser_job],
                    outputs=[identifier, status, stage, progress, download, preview],
                    queue=False, show_progress="hidden", api_name=False,
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

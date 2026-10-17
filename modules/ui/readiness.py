"""On-device first-run readiness and opt-in Hugging Face token setup.

Read-only checks do not download models or promise successful inference.
The model weights are downloaded by the existing WhisperX pipeline on first use.
Never include token values, full paths or exception messages in UI output.
"""
from __future__ import annotations

import os
from pathlib import Path
import re

from dotenv import load_dotenv, set_key

ROOT = Path(__file__).resolve().parents[2]
HF_MODELS = (
    "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "https://huggingface.co/pyannote/segmentation-3.0",
)
REQUIRED_PACKAGES = (
    "torch", "torchaudio", "whisperx", "pyannote.audio",
    "gradio", "faster-whisper", "ctranslate2",
)


def load_local_environment() -> None:
    """Import an existing private .env without overwriting process variables."""
    env_path = ROOT / ".env"
    if env_path.is_file() and not env_path.is_symlink():
        load_dotenv(env_path, override=False)


def save_hf_token(value: str) -> str:
    """Store explicitly submitted HF credentials in Git-ignored .env.

    This is a local plaintext secrets file. It is not committed, shared,
    logged, included in the transcript or returned to the browser.
    """
    if not isinstance(value, str):
        return "Токен не сохранён: вставьте токен Hugging Face."
    token = value.strip()
    if not re.fullmatch(r"hf_[A-Za-z0-9]{8,}", token):
        return "Токен не сохранён: ожидается токен Hugging Face вида hf_..."
    destination = ROOT / ".env"
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        return "Токен не сохранён: небезопасный путь локальных настроек."
    try:
        if not destination.exists():
            destination.touch(mode=0o600)
        # set_key retains unrelated .env settings. The value never enters Git.
        outcome = set_key(str(destination), "HF_TOKEN", token, quote_mode="always")
        if not outcome or not outcome[0]:
            return "Токен не сохранён: не удалось обновить локальные настройки."
    except (OSError, ValueError):
        return "Токен не сохранён: нет доступа к файлу настроек."
    os.environ["HF_TOKEN"] = token
    return (
        "Токен сохранён только на этом ПК. Примите условия двух моделей "
        "Pyannote на Hugging Face, затем нажмите «Проверить готовность»."
    )


def readiness_message() -> str:
    """Report basic readiness, never a synthetic 'model passed' verdict."""
    load_local_environment()
    from modules.utils.environment_preflight import build_report

    try:
        report = build_report(check_hf=True, skip_torch=False)
    except (OSError, ValueError, RuntimeError):
        return "Проверка не завершилась. Посмотрите локальный журнал запуска."

    problems: list[str] = []
    if not report["python"].get("supported_by_readme"):
        problems.append("Python должен быть версии 3.10–3.12.")
    for tool, label in (("ffmpeg", "FFmpeg"), ("ffprobe", "FFprobe")):
        if report[tool].get("status") != "ok":
            problems.append(f"{label} не найден в PATH.")
    for name in REQUIRED_PACKAGES:
        if report["packages"].get(name, {}).get("status") != "installed":
            problems.append(f"Не установлен пакет {name}.")
    if report["torch"].get("status") != "ok":
        problems.append("PyTorch не импортируется корректно.")

    hf = report["hugging_face"]
    if not hf.get("token_present"):
        problems.append("Нет токена Hugging Face для моделей Pyannote.")
    elif hf.get("authentication") == "failed":
        problems.append("Hugging Face не принял токен.")
    elif hf.get("authentication") != "verified":
        problems.append(
            "Не удалось проверить токен Hugging Face онлайн. "
            "Офлайн-кэш моделей не проверен."
        )

    if problems:
        return (
            "Нужна первоначальная настройка:\n"
            + "\n".join(f"• {item}" for item in problems)
            + "\nПосле устранения причин нажмите «Проверить готовность» ещё раз."
        )

    result = (
        "Основные компоненты найдены; токен принят Hugging Face. "
        "Можно выбрать аудиофайл и запустить первую обработку.\n"
        "При первом запуске WhisperX сам скачает отсутствующие модели "
        "в локальный кэш. Доступ к обоим файлам Pyannote и реальная "
        "работа распознавания ещё НЕ подтверждены этой проверкой."
    )
    memory = report.get("ram", {})
    if memory.get("status") == "ok" and memory.get("total_gib", 100) < 12:
        result += (
            "\nВнимание: на компьютере менее 12 ГБ RAM. "
            "Whisper Small + Pyannote могут не поместиться в память."
        )
    return result

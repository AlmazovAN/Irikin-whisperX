"""Local first-run readiness is transparent, never a fake model smoke-test."""
from __future__ import annotations

from pathlib import Path

import pytest

from modules.ui import readiness
from modules.utils import environment_preflight

pytestmark = pytest.mark.fast


def mock_report(*, good: bool, token: str = "verified", low_ram: bool = False):
    packages = {
        name: {"status": "installed" if good else "missing"}
        for name in readiness.REQUIRED_PACKAGES
    }
    return {
        "python": {"supported_by_readme": good},
        "ffmpeg": {"status": "ok" if good else "missing"},
        "ffprobe": {"status": "ok" if good else "missing"},
        "packages": packages,
        "torch": {"status": "ok" if good else "error"},
        "hugging_face": {
            "token_present": token != "missing",
            "authentication": token if token != "missing" else "not_checked",
        },
        "ram": {"status": "ok", "total_gib": 8 if low_ram else 16},
    }


def test_ready_prerequisites_do_not_claim_real_asr_success(monkeypatch):
    monkeypatch.setattr(readiness, "load_local_environment", lambda: None)
    seen = []
    monkeypatch.setattr(environment_preflight, "build_report",
                        lambda **kwargs: (seen.append(kwargs), mock_report(good=True))[1])
    result = readiness.readiness_message()
    assert "Можно выбрать аудиофайл" in result
    assert "НЕ подтверждены" in result
    assert seen == [{"check_hf": True, "skip_torch": False}]


def test_missing_packages_and_hf_are_actionable_and_not_ready(monkeypatch):
    monkeypatch.setattr(readiness, "load_local_environment", lambda: None)
    monkeypatch.setattr(environment_preflight, "build_report",
                        lambda **kwargs: mock_report(good=False, token="missing"))
    result = readiness.readiness_message()
    assert "Нужна первоначальная настройка" in result
    assert "FFmpeg" in result and "Python" in result and "Hugging Face" in result
    assert "Можно выбрать аудиофайл" not in result


def test_hf_offline_and_insufficient_ram_are_not_hidden(monkeypatch):
    monkeypatch.setattr(readiness, "load_local_environment", lambda: None)
    monkeypatch.setattr(environment_preflight, "build_report",
                        lambda **kwargs: mock_report(good=True, token="not_checked"))
    assert "Не удалось проверить токен" in readiness.readiness_message()
    monkeypatch.setattr(environment_preflight, "build_report",
                        lambda **kwargs: mock_report(good=True, low_ram=True))
    assert "менее 12 ГБ RAM" in readiness.readiness_message()


def test_invalid_token_does_not_create_file_or_echo_token(monkeypatch, tmp_path):
    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    secret = "invalid-token"
    message = readiness.save_hf_token(secret)
    assert "не сохранён" in message
    assert secret not in message
    assert not (tmp_path / ".env").exists()


def test_token_is_saved_locally_and_other_env_entries_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text("KEEP_THIS=1\n", encoding="utf-8")
    token = "hf_" + "AbC123456789"
    message = readiness.save_hf_token(token)
    assert "сохранён" in message
    assert token not in message
    assert "KEEP_THIS" in env.read_text(encoding="utf-8")
    assert token in env.read_text(encoding="utf-8")
    assert readiness.os.environ["HF_TOKEN"] == token
    monkeypatch.delenv("HF_TOKEN")
    readiness.load_local_environment()
    assert readiness.os.environ["HF_TOKEN"] == token


def test_symlinked_env_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    external = tmp_path / "elsewhere"
    external.write_text("SAFE=1\n", encoding="utf-8")
    destination = tmp_path / ".env"
    try:
        destination.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this host")
    message = readiness.save_hf_token("hf_ABC123456789")
    assert "не сохранён" in message
    assert external.read_text(encoding="utf-8") == "SAFE=1\n"

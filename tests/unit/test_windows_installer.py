"""Offline installer tests: no real subprocess, pip install, or venv creation."""

import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from scripts import install_windows as installer

pytestmark = pytest.mark.fast


def test_two_step_install_is_explicitly_index_scoped():
    commands = installer.install_steps(Path("C:/app/venv/Scripts/python.exe"), "cpu")
    assert len(commands) == 3
    assert commands[0][commands[0].index("--index-url") + 1].endswith("/cpu")
    assert "--index-url" not in commands[1]
    assert "requirements.txt" in commands[1][-1]
    assert commands[-1][-1] == "check"


def test_cuda_has_explicit_pytorch_index():
    commands = installer.install_steps(Path("C:/a/venv/Scripts/python.exe"), "cuda126")
    assert "cu126" in commands[0]


def test_dry_run_does_not_execute_or_modify(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    monkeypatch.setattr(installer, "VENV", tmp_path / "venv")
    def forbid(*a, **kw):
        raise AssertionError("subprocess should not run on dry-run")
    monkeypatch.setattr(installer.subprocess, "run", forbid)
    assert installer.execute("cpu", dry_run=True, host_os="posix") == 0
    assert not (tmp_path / "venv").exists()
    assert "DRY RUN" in capsys.readouterr().out


def test_regular_install_refuses_non_windows(monkeypatch):
    def forbid(*a, **kw):
        raise AssertionError("no installation on non-Windows")
    monkeypatch.setattr(installer.subprocess, "run", forbid)
    assert installer.execute("cpu", host_os="posix", py_version=(3, 11)) == 2


def test_existing_unmanaged_venv_is_protected(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    assert "back" in installer.existing_venv_error(
        "cpu", reuse_venv=True, venv=venv, marker=venv / "profile.json"
    ).lower()
    assert "venv already exists" in installer.existing_venv_error(
        "cpu", reuse_venv=False, venv=venv, marker=venv / "profile.json"
    )


def test_reuse_requires_matching_profile(tmp_path):
    venv = tmp_path / "venv"
    (venv / "Scripts").mkdir(parents=True)
    (venv / "Scripts" / "python.exe").write_bytes(b"fake")
    marker = venv / ".irikin-install-profile.json"
    marker.write_text(json.dumps({"profile": "cpu", "status": "incomplete"}), encoding="utf-8")
    assert installer.existing_venv_error("cpu", reuse_venv=True, venv=venv, marker=marker) is None
    assert installer.existing_venv_error("cuda126", reuse_venv=True, venv=venv, marker=marker)


def test_nvidia_guard_handles_missing_tool(monkeypatch):
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)
    assert installer.nvidia_available() is False


def test_nvidia_guard_rejects_failed_command(monkeypatch):
    monkeypatch.setattr(installer.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout=""))
    assert installer.nvidia_available() is False


def test_nvidia_guard_accepts_listed_device(monkeypatch):
    monkeypatch.setattr(installer.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="GPU 0: NVIDIA Test"))
    assert installer.nvidia_available() is True


def test_unsupported_python_is_not_installed(monkeypatch):
    monkeypatch.setattr(installer, "nvidia_available", lambda: True)
    assert installer.execute("cpu", host_os="nt", py_version=(3, 13)) == 2


def test_requirements_pin_matching_torch_trio():
    root = Path(__file__).resolve().parents[2]
    text = (root / "requirements" / "torch.txt").read_text(encoding="utf-8")
    assert re.search(r"(?m)^torch==2\.7\.1$", text)
    assert re.search(r"(?m)^torchaudio==2\.7\.1$", text)
    assert re.search(r"(?m)^torchvision==0\.22\.1$", text)


def test_no_extra_index_or_duplicate_openai_whisper():
    root = Path(__file__).resolve().parents[2]
    text = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "--extra-index-url" not in text
    names = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-r "):
            continue
        name = re.split(r"\s*@\s*|[<>=!~\[]", line, maxsplit=1)[0]
        names.append(re.sub(r"[-_.]+", "-", name).lower())
    assert len(names) == len(set(names))
    assert names.count("openai-whisper") == 1
    assert "onnxruntime-gpu" not in names


def test_installer_entrypoint_uses_python_script():
    root = Path(__file__).resolve().parents[2]
    text = (root / "Install.bat").read_text(encoding="utf-8")
    assert "install_windows.py" in text
    assert "exit /b %INSTALL_RESULT%" in text

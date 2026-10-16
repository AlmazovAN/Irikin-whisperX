"""Conservative Windows installer for the legacy WhisperX WebUI.

This script intentionally never deletes an existing venv and does not claim
that package metadata resolution proves audio inference compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / "venv"
MARKER = VENV / ".irikin-install-profile.json"
PYTORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda126": "https://download.pytorch.org/whl/cu126",
}
TORCH_REQUIREMENTS = ROOT / "requirements" / "torch.txt"
ALL_REQUIREMENTS = ROOT / "requirements.txt"
SUPPORTED_PYTHON = {(3, 10), (3, 11), (3, 12)}


def venv_python(root: Path = ROOT) -> Path:
    return root / "venv" / "Scripts" / "python.exe"


def install_steps(python: Path, profile: str) -> list[list[str]]:
    """Never combine PyTorch's wheel index with the full PyPI dependency graph."""
    pip = [str(python), "-m", "pip", "--disable-pip-version-check", "--no-input"]
    return [
        [*pip, "install", "--index-url", PYTORCH_INDEXES[profile],
         "-r", str(TORCH_REQUIREMENTS)],
        [*pip, "install", "-r", str(ALL_REQUIREMENTS)],
        [*pip, "check"],
    ]


def nvidia_available() -> bool:
    """Guard against accidentally choosing CUDA on an AMD-only computer."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        result = subprocess.run(
            [executable, "-L"], capture_output=True, text=True,
            errors="replace", timeout=10, check=False,
        )
        return result.returncode == 0 and "GPU" in result.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def read_marker(marker: Path = MARKER) -> dict | None:
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_marker(profile: str, status: str, marker: Path = MARKER) -> None:
    data = {"profile": profile, "status": status, "torch_version": "2.7.1"}
    temporary = marker.with_name(marker.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)


def existing_venv_error(profile: str, *, reuse_venv: bool,
                        venv: Path = VENV, marker: Path = MARKER) -> str | None:
    """Never overwrite a prior user environment or switch binary profiles."""
    if not venv.exists():
        return None
    stored = read_marker(marker)
    if not reuse_venv:
        return (
            "venv already exists. Installation stopped to protect the working environment. "
            "Use --reuse-venv only if it was created by this installer, or back it up manually."
        )
    if not stored or stored.get("profile") != profile:
        return (
            "Existing venv has no matching P03 profile marker. "
            "Back up/rename it manually; this installer will not replace or delete it."
        )
    if not (venv / "Scripts" / "python.exe").is_file():
        return "Existing venv has no Scripts/python.exe; rename or repair it manually."
    return None


def execute(profile: str, *, dry_run: bool = False, reuse_venv: bool = False,
            host_os: str | None = None, py_version: tuple[int, int] | None = None) -> int:
    os_name = host_os or os.name
    version = py_version or sys.version_info[:2]
    if profile not in PYTORCH_INDEXES:
        print(f"Unsupported profile: {profile}", file=sys.stderr)
        return 2

    target = venv_python()
    print(f"Profile: {profile} (PyTorch 2.7.1 / torchaudio 2.7.1)")
    if dry_run:
        print("DRY RUN: no environment, packages or model files will be changed.")
        for step in install_steps(target, profile):
            print("COMMAND:", subprocess.list2cmdline(step))
        return 0

    if os_name != "nt":
        print("This installer targets Windows only. No changes made.", file=sys.stderr)
        return 2
    if version not in SUPPORTED_PYTHON or sys.maxsize <= 2**32:
        print("Use 64-bit CPython 3.10–3.12 (Python 3.11 recommended).", file=sys.stderr)
        return 2
    if not TORCH_REQUIREMENTS.is_file() or not ALL_REQUIREMENTS.is_file():
        print("Missing requirements. Run from a complete repository checkout.", file=sys.stderr)
        return 2
    if profile == "cuda126" and not nvidia_available():
        print("NVIDIA GPU/driver not found. CUDA installation refused; choose --profile cpu.", file=sys.stderr)
        return 2

    issue = existing_venv_error(profile, reuse_venv=reuse_venv)
    if issue:
        print(issue, file=sys.stderr)
        return 2

    if not VENV.exists():
        print("Creating isolated virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(VENV)], cwd=ROOT, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"Failed to create venv: {type(exc).__name__}", file=sys.stderr)
            return 1
        if not target.is_file():
            print("venv creation failed: Scripts/python.exe not found.", file=sys.stderr)
            return 1
    try:
        write_marker(profile, "installing")
        for command in install_steps(target, profile):
            print("Running:", subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        write_marker(profile, "installed")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(
            f"Installation failed: {type(exc).__name__}. "
            "The existing venv was not removed. Fix the error and use "
            "--reuse-venv to retry the same profile.", file=sys.stderr,
        )
        return 1
    print("Dependencies installed; pip check passed. ASR/Pyannote are NOT yet validated.")
    print("Run python scripts/diagnose_environment.py from the new venv, then P06 smoke-test.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install Irikin-whisperX on Windows without replacing an existing venv"
    )
    parser.add_argument("--profile", choices=sorted(PYTORCH_INDEXES), default="cpu",
                        help="CPU default (AMD/iGPU); CUDA 12.6 only for supported NVIDIA GPUs")
    parser.add_argument("--dry-run", action="store_true", help="Show planned commands without changes")
    parser.add_argument("--reuse-venv", action="store_true",
                        help="Retry/update only a venv with a matching P03 profile marker")
    args = parser.parse_args(argv)
    return execute(profile=args.profile, dry_run=args.dry_run, reuse_venv=args.reuse_venv)


if __name__ == "__main__":
    raise SystemExit(main())

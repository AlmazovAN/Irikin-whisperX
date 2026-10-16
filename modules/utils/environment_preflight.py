"""Read-only, dependency-light environment probes for the local WhisperX app.

All probes are best-effort. They never install packages, download models or
report that an inference pipeline works without an actual transcription test.
"""

from __future__ import annotations

import ctypes
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any
import urllib.error
import urllib.request

SUPPORTED_PYTHON = {(3, 10), (3, 11), (3, 12)}
PACKAGES = (
    "torch", "torchaudio", "whisperx", "pyannote.audio", "faster-whisper",
    "ctranslate2", "gradio", "transformers", "huggingface-hub", "numpy",
)
HF_HOST = "https://huggingface.co/api/whoami-v2"


def probe_command(command: str) -> dict[str, Any]:
    executable = shutil.which(command)
    if not executable:
        return {"status": "missing", "detail": f"{command} not found in PATH"}
    try:
        result = subprocess.run(
            [executable, "-version"], capture_output=True, text=True,
            errors="replace", timeout=7, check=False,
        )
        if result.returncode != 0:
            return {"status": "error", "detail": f"{command} exited with code {result.returncode}"}
        first_line = result.stdout.splitlines()
        return {"status": "ok", "version": (first_line[0][:200] if first_line else "unknown")}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "detail": f"{command} could not run: {type(exc).__name__}"}


def probe_windows_memory() -> dict[str, Any]:
    if platform.system() != "Windows":
        return {"status": "not_applicable", "detail": "Windows-only physical memory probe"}

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return {
            "status": "ok", "total_gib": round(status.ullTotalPhys / (1024**3), 2),
            "available_gib": round(status.ullAvailPhys / (1024**3), 2),
        }
    except (AttributeError, OSError, ValueError) as exc:
        return {"status": "error", "detail": f"Windows RAM query: {type(exc).__name__}"}


def probe_windows_gpus() -> dict[str, Any]:
    if platform.system() != "Windows":
        return {"status": "not_applicable", "adapters": [], "detail": "Windows GPU enumeration only"}
    shell = shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        return {"status": "unknown", "adapters": [], "detail": "PowerShell not available"}
    statement = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion,Status | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-Command", statement],
            capture_output=True, text=True, errors="replace", timeout=12, check=False,
        )
        if result.returncode != 0:
            return {"status": "unknown", "adapters": [], "detail": "Windows GPU query failed"}
        raw = json.loads(result.stdout.strip().lstrip("\ufeff"))
        devices = raw if isinstance(raw, list) else [raw]
        adapters = [
            {
                "name": str(device.get("Name") or "unknown"),
                "driver_version": device.get("DriverVersion"),
                "reported_adapter_ram_bytes": device.get("AdapterRAM"),
                "vram_reliable": False,
            }
            for device in devices if isinstance(device, dict)
        ]
        return {
            "status": "ok" if adapters else "unknown", "adapters": adapters,
            "detail": "Win32 AdapterRAM may be inaccurate (32-bit value, shared/integrated GPUs)",
        }
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {"status": "unknown", "adapters": [], "detail": "Cannot parse Windows GPU data"}


def probe_nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"status": "not_available", "devices": []}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, errors="replace", timeout=8, check=False,
        )
        if result.returncode != 0:
            return {"status": "unknown", "devices": [], "detail": "nvidia-smi returned an error"}
        devices = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                memory_mib = int(parts[1])
            except ValueError:
                continue
            devices.append({"name": parts[0], "vram_mib": memory_mib, "driver_version": parts[2]})
        return {"status": "ok" if devices else "unknown", "devices": devices}
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "unknown", "devices": [], "detail": "nvidia-smi unavailable"}


def probe_packages() -> dict[str, Any]:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = {"status": "installed", "version": metadata.version(name)}
        except metadata.PackageNotFoundError:
            packages[name] = {"status": "missing"}
        except Exception as exc:
            packages[name] = {"status": "error", "detail": type(exc).__name__}
    return packages


def probe_torch(packages: dict[str, Any], *, skip: bool = False) -> dict[str, Any]:
    if skip:
        return {"status": "not_checked", "detail": "Skipped by --skip-torch"}
    if packages.get("torch", {}).get("status") != "installed":
        return {"status": "missing", "detail": "Install compatible torch in P03"}
    try:
        torch = importlib.import_module("torch")
        cuda = torch.cuda
        enabled = bool(cuda.is_available())
        devices = []
        if enabled:
            for index in range(cuda.device_count()):
                properties = cuda.get_device_properties(index)
                devices.append({
                    "name": str(properties.name),
                    "vram_mib": round(properties.total_memory / (1024**2)),
                })
        xpu = getattr(torch, "xpu", None)
        return {
            "status": "ok", "cuda_available": enabled,
            "torch_cuda_build": getattr(torch.version, "cuda", None),
            "cuda_devices": devices,
            "xpu_available": bool(xpu.is_available()) if xpu is not None else False,
            "detail": "CUDA false does not rule out CPU execution; AMD iGPU is not NVIDIA CUDA",
        }
    except Exception as exc:
        return {
            "status": "error", "detail": f"torch import/device check failed: {type(exc).__name__}",
        }


def probe_hf(*, check_online: bool = False) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return {
            "status": "missing_token", "token_present": False,
            "authentication": "not_checked", "model_access": "not_checked",
            "detail": "HF_TOKEN or HUGGING_FACE_HUB_TOKEN is absent; gated Pyannote may require it",
        }
    if not check_online:
        return {
            "status": "token_present", "token_present": True,
            "authentication": "not_checked", "model_access": "not_checked",
            "detail": "Token exists; validity and Pyannote license acceptance are unverified",
        }
    request = urllib.request.Request(
        HF_HOST, headers={"Authorization": "Bearer " + token, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            if response.status != 200:
                return {
                    "status": "auth_failed", "token_present": True,
                    "authentication": "failed", "model_access": "not_checked",
                    "detail": f"Hugging Face returned HTTP {response.status}",
                }
        return {
            "status": "authenticated", "token_present": True,
            "authentication": "verified", "model_access": "not_checked",
            "detail": "Identity accepted by HF; gated model licenses still need a real Pyannote test",
        }
    except urllib.error.HTTPError as exc:
        return {
            "status": "auth_failed", "token_present": True,
            "authentication": "failed", "model_access": "not_checked",
            "detail": f"Hugging Face returned HTTP {exc.code}",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "status": "network_error", "token_present": True,
            "authentication": "not_checked", "model_access": "not_checked",
            "detail": f"HF connectivity failed: {type(exc).__name__}",
        }


def build_report(*, check_hf: bool = False, skip_torch: bool = False,
                 disk_path: str | Path | None = None) -> dict[str, Any]:
    """Collect observations only. No model loading or installation happens here."""
    python = {
        "version": platform.python_version(),
        "supported_by_readme": sys.version_info[:2] in SUPPORTED_PYTHON,
        "executable_arch": platform.architecture()[0],
    }
    system = {
        "name": platform.system(), "release": platform.release(),
        "version": platform.version(), "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
    root = Path(disk_path) if disk_path else Path(__file__).resolve().parents[2]
    try:
        disk = shutil.disk_usage(root)
        storage = {
            "status": "ok", "free_gib": round(disk.free / (1024**3), 2),
            "total_gib": round(disk.total / (1024**3), 2),
        }
    except OSError as exc:
        storage = {"status": "error", "detail": f"Disk check failed: {type(exc).__name__}"}
    packages = probe_packages()
    report = {
        "schema_version": 1,
        "purpose": "Non-destructive environment diagnostic; NOT a successful inference test",
        "system": system,
        "python": python,
        "ram": probe_windows_memory(),
        "disk": storage,
        "ffmpeg": probe_command("ffmpeg"),
        "ffprobe": probe_command("ffprobe"),
        "windows_gpus": probe_windows_gpus(),
        "nvidia_smi": probe_nvidia_smi(),
        "packages": packages,
        "torch": probe_torch(packages, skip=skip_torch),
        "hugging_face": probe_hf(check_online=check_hf),
    }
    blockers = []
    warnings = []
    if system["name"] != "Windows":
        warnings.append("Target is Windows 11; this host has a different OS")
    if not python["supported_by_readme"]:
        blockers.append("Python version is outside the currently documented range 3.10–3.12")
    for command in ("ffmpeg", "ffprobe"):
        if report[command]["status"] != "ok":
            blockers.append(f"{command} is not available: {report[command].get('detail', 'unknown failure')}")
    if packages["torch"]["status"] != "installed":
        blockers.append("PyTorch is not installed; inference cannot run in this environment")
    elif report["torch"]["status"] == "error":
        blockers.append(report["torch"]["detail"])
    if packages["whisperx"]["status"] != "installed":
        blockers.append("WhisperX is not installed; inference cannot run in this environment")
    if packages["pyannote.audio"]["status"] != "installed":
        warnings.append("Pyannote is missing; speaker diarization cannot run")
    if report["hugging_face"]["status"] == "missing_token":
        warnings.append("Hugging Face token not found; gated diarization may be blocked")
    if report["torch"]["status"] == "ok" and not report["torch"]["cuda_available"]:
        warnings.append("No CUDA device is available to torch; CPU inference may be slow")
    if storage.get("status") == "ok" and storage["free_gib"] < 5:
        warnings.append("Less than 5 GiB free; model downloads may require more disk space")
    if report["ram"].get("status") == "ok" and report["ram"]["total_gib"] < 12:
        warnings.append("Less than 12 GiB system RAM; Small + Pyannote may be memory constrained")
    report["assessment"] = {
        "status": "blocked_prerequisites" if blockers else "preflight_checks_completed",
        "blockers": blockers, "warnings": warnings,
        "inference_validated": False,
    }
    return report


def render_text(report: dict[str, Any]) -> str:
    """Human-readable summary. Do not include usernames, paths, or credentials."""
    system, python = report["system"], report["python"]
    lines = [
        "WhisperX environment preflight (no models downloaded)",
        f"OS: {system['name']} {system['release']} ({system['machine']})",
        f"Python: {python['version']} (README-supported: {python['supported_by_readme']})",
        f"RAM: {report['ram'].get('total_gib', 'unknown')} GiB total",
        f"Disk: {report['disk'].get('free_gib', 'unknown')} GiB free",
        f"FFmpeg: {report['ffmpeg'].get('version', report['ffmpeg'].get('detail', 'unknown'))}",
        f"FFprobe: {report['ffprobe'].get('version', report['ffprobe'].get('detail', 'unknown'))}",
        f"Windows video adapters: {', '.join(d['name'] for d in report['windows_gpus'].get('adapters', [])) or 'not confirmed'}",
        f"NVIDIA GPUs: {', '.join(d['name'] for d in report['nvidia_smi'].get('devices', [])) or 'not detected'}",
        f"Torch CUDA: {report['torch'].get('cuda_available', 'not confirmed')}",
        f"HF: {report['hugging_face']['status']} (gated model access not verified)",
        "Installed: " + ", ".join(
            f"{name}={info.get('version', 'missing')}" for name, info in report["packages"].items()
        ),
        "Status: " + report["assessment"]["status"],
    ]
    for label, items in (("BLOCKER", report["assessment"]["blockers"]),
                         ("WARNING", report["assessment"]["warnings"])):
        lines.extend(f"{label}: {item}" for item in items)
    lines.append("NOTE: This report does not prove WhisperX ASR/alignment/diarization works.")
    return "\n".join(lines)

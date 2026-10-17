"""Single-process Windows launcher for the local Gradio WebUI.

No dependency installation, model download, backend server, or async job queue.
Keeps the console attached to the one Gradio server until it exits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
PREFLIGHT = ROOT / "scripts" / "diagnose_environment.py"
APP = ROOT / "app.py"
LOGS = ROOT / "logs"
HOST = "127.0.0.1"
REQUIRED_PACKAGES = ("torch", "torchaudio", "whisperx", "gradio", "pyannote.audio")


class LaunchError(RuntimeError):
    """Actionable startup failure, not an ASR failure."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Windows WhisperX launcher")
    parser.add_argument("--port", "--server_port", type=int, default=7860,
                        help="Local Gradio port (default: 7860)")
    parser.add_argument("--check-only", action="store_true",
                        help="Check environment and port without starting server")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open browser after server is ready")
    parser.add_argument("--startup-timeout", type=int, default=180,
                        help="Wait for first local HTTP response, in seconds (15–900)")
    parser.add_argument("--ui-mode", "--ui_mode", choices=("simple", "advanced"),
                        default="simple",
                        help="Default Simple job queue, or legacy Advanced interface")
    parser.add_argument("--whisper_type", choices=("whisper", "faster-whisper", "insanely-fast-whisper", "whisperx"),
                        help="Legacy inference engine override")
    parser.add_argument("--theme", help="Legacy Gradio theme name")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    if not 15 <= args.startup_timeout <= 900:
        parser.error("--startup-timeout must be between 15 and 900")
    return args


def local_url(port: int) -> str:
    return f"http://{HOST}:{port}/"


def port_is_occupied(port: int) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout=0.5):
            return True
    except (ConnectionRefusedError, TimeoutError):
        return False
    except OSError as exc:
        # An unexpected socket error is not proof of a free port.
        raise LaunchError(f"Unable to check port {port}: {type(exc).__name__}") from exc


def build_app_command(python: Path, args: argparse.Namespace) -> list[str]:
    command = [str(python), "-u", str(APP), "--server_name", HOST,
               "--server_port", str(args.port), "--share", "false",
               "--api_open", "false", "--inbrowser", "false",
               "--ui_mode", args.ui_mode]
    if args.whisper_type:
        command.extend(("--whisper_type", args.whisper_type))
    if args.theme:
        command.extend(("--theme", args.theme))
    return command


def check_venv(root: Path = ROOT, python: Path | None = None) -> list[str]:
    """Read-only metadata preflight; do not import torch or any model."""
    target = python if python is not None else root / "venv" / "Scripts" / "python.exe"
    preflight = root / "scripts" / "diagnose_environment.py"
    app = root / "app.py"
    if not app.is_file() or not preflight.is_file():
        raise LaunchError("Incomplete checkout: app.py or diagnostic script is missing.")
    if not target.is_file():
        raise LaunchError("Virtual environment not found. Run Install.bat --profile cpu first.")
    try:
        probe = subprocess.run(
            [str(target), str(preflight), "--json", "--skip-torch"],
            cwd=root, capture_output=True, text=True, errors="replace",
            timeout=45, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LaunchError(f"Environment check failed: {type(exc).__name__}") from exc
    if probe.returncode != 0:
        raise LaunchError("Environment check could not run. Verify the venv Python and dependencies.")
    try:
        report = json.loads(probe.stdout)
        if not isinstance(report, dict):
            raise ValueError("invalid report")
        assessment = report["assessment"]
        packages = report["packages"]
        issues = list(assessment["blockers"])
        for name in REQUIRED_PACKAGES:
            if packages.get(name, {}).get("status") != "installed":
                message = f"Missing required package: {name}"
                if message not in issues:
                    issues.append(message)
        if issues:
            raise LaunchError("Preflight blocked: " + "; ".join(issues))
        return list(assessment.get("warnings", []))
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise LaunchError(f"Cannot read environment report: {type(exc).__name__}") from exc


def http_is_ready(port: int, *, opener=None) -> bool:
    """Local HTTP readiness, without system HTTP_PROXY/HTTPS_PROXY settings."""
    if opener is None:
        opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(Request(local_url(port), headers={"User-Agent": "Irikin-WhisperX-Launcher"}), timeout=1) as response:
            return 200 <= response.status < 400
    except (URLError, HTTPError, OSError, TimeoutError):
        return False


@contextmanager
def launch_lock(log_dir: Path):
    """Nonblocking Windows byte lock, preventing two launchers racing for the port."""
    log_dir.mkdir(parents=True, exist_ok=True)
    lock_path = log_dir / "launcher.lock"
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise LaunchError("Another WhisperX launcher is already running.") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


class RunLog:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = log_dir / f"start-{timestamp}-{os.getpid()}.log"
        self._file = self.path.open("x", encoding="utf-8", errors="replace")
        self._lock = threading.Lock()

    def write(self, message: str) -> None:
        with self._lock:
            print(message, flush=True)
            self._file.write(message + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


def stream_server_output(process: subprocess.Popen, log: RunLog) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        log.write("[server] " + line.rstrip("\r\n"))


def stop_server(process: subprocess.Popen, log: RunLog, *, grace_seconds: int = 10) -> None:
    if process.poll() is not None:
        return
    log.write("Stopping local server...")
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
        process.wait(timeout=grace_seconds)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            log.write("Server did not stop cleanly; terminating its process.")
            process.kill()
            process.wait(timeout=5)


def run(args: argparse.Namespace, *, root: Path = ROOT) -> int:
    log_dir = root / "logs"
    with launch_lock(log_dir):
        log = RunLog(log_dir)
        child = None
        reader = None
        try:
            log.write("WhisperX local launcher (one Gradio process; no separate backend)")
            log.write(f"Log file: {log.path}")
            log.write(f"Local address: {local_url(args.port)}")
            log.write("Checking Windows venv, FFmpeg and package metadata (no model download)...")
            for warning in check_venv(root):
                log.write("WARNING: " + warning)
            if port_is_occupied(args.port):
                raise LaunchError(f"Port {args.port} is already in use. Close the other server or use --port another_number.")
            if args.check_only:
                log.write("CHECK OK: prerequisites found. Actual ASR/diarization is not validated.")
                return 0
            log.write("Starting app.py. Startup may be slow the first time.")
            command = build_app_command(root / "venv" / "Scripts" / "python.exe", args)
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            child = subprocess.Popen(
                command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=flags,
            )
            reader = threading.Thread(target=stream_server_output, args=(child, log), daemon=True)
            reader.start()
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise LaunchError(f"Server exited during startup (code {child.returncode}); check log above.")
                if http_is_ready(args.port):
                    log.write(f"READY: {local_url(args.port)}")
                    if not args.no_browser:
                        try:
                            webbrowser.open_new_tab(local_url(args.port))
                        except Exception as exc:
                            log.write(f"Browser could not open ({type(exc).__name__}); copy the local URL above.")
                    log.write("Keep this console open. Press Ctrl+C to stop the server.")
                    break
                time.sleep(0.4)
            else:
                raise LaunchError(f"Server did not respond within {args.startup_timeout}s; check log above.")
            return child.wait()
        except KeyboardInterrupt:
            log.write("Stop requested from keyboard.")
            return 0
        except (LaunchError, OSError) as exc:
            log.write(f"ERROR: {exc}")
            return 2
        finally:
            if child is not None:
                stop_server(child, log)
                if reader is not None:
                    reader.join(timeout=2)
            log.write("Launcher finished.")
            log.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if os.name != "nt":
        print("Windows-only launcher. No server started.", file=sys.stderr)
        return 2
    try:
        return run(args)
    except (LaunchError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

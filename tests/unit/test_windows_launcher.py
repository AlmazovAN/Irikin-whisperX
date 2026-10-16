"""P05 launcher safety checks: no Gradio, models, network or pip needed."""

import importlib.util
from io import StringIO
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "launch_windows.py"
spec = importlib.util.spec_from_file_location("whisperx_windows_launcher", SOURCE)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

pytestmark = pytest.mark.fast


def args(*argv):
    return launcher.parse_args(list(argv))


def test_defaults_stay_local():
    a = args()
    assert a.port == 7860 and not a.no_browser and not a.check_only
    assert launcher.local_url(a.port) == "http://127.0.0.1:7860/"
    command = launcher.build_app_command(Path("C:/repo/venv/python.exe"), a)
    for setting, value in (("--server_name", "127.0.0.1"),
                           ("--share", "false"), ("--api_open", "false"),
                           ("--inbrowser", "false"), ("--server_port", "7860")):
        assert command[command.index(setting) + 1] == value


@pytest.mark.parametrize("bad", ["--share", "--api_open", "--server_name", "--allowed_paths", "--password"])
def test_launcher_rejects_unsafe_and_unsupported_flags(bad):
    with pytest.raises(SystemExit):
        args(bad, "true")


@pytest.mark.parametrize("port", ["0", "80", "65536", "-1", "not_a_port"])
def test_invalid_ports_rejected(port):
    with pytest.raises(SystemExit):
        args("--port", port)


def test_port_alias_and_options():
    a = args("--server_port", "8999", "--no-browser", "--startup-timeout", "300")
    assert a.port == 8999 and a.no_browser and a.startup_timeout == 300


def test_no_venv_preflight_is_actionable(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "diagnose_environment.py").write_text("")
    (tmp_path / "app.py").write_text("")
    with pytest.raises(launcher.LaunchError, match="Install.bat"):
        launcher.check_venv(tmp_path)


def fake_install(tmp_path):
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "diagnose_environment.py").write_text("")
    (tmp_path / "app.py").write_text("")
    python = tmp_path / "venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("")
    return python


def report(*, blockers=(), gradio=True):
    packages = {name: {"status": "installed"} for name in launcher.REQUIRED_PACKAGES}
    if not gradio:
        packages["gradio"] = {"status": "missing"}
    return {"assessment": {"blockers": list(blockers), "warnings": ["CPU is slow"]},
            "packages": packages}


def test_preflight_reads_report_without_importing_torch(monkeypatch, tmp_path):
    fake_install(tmp_path)
    seen = {}
    def run(command, **kwargs):
        seen["command"] = command
        return SimpleNamespace(returncode=0, stdout=json.dumps(report()))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.check_venv(tmp_path) == ["CPU is slow"]
    assert "--skip-torch" in seen["command"]
    assert "--json" in seen["command"]


@pytest.mark.parametrize("issue", ["blocker", "missing"])
def test_preflight_stops_on_missing_dependencies(monkeypatch, tmp_path, issue):
    fake_install(tmp_path)
    data = report(blockers=["ffmpeg unavailable"] if issue == "blocker" else [],
                  gradio=issue != "missing")
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(data)))
    with pytest.raises(launcher.LaunchError, match="Preflight blocked"):
        launcher.check_venv(tmp_path)


def test_preflight_handles_bad_json(monkeypatch, tmp_path):
    fake_install(tmp_path)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="not json"))
    with pytest.raises(launcher.LaunchError, match="Cannot read environment report"):
        launcher.check_venv(tmp_path)


def test_occupied_port_refuses_to_start(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "check_venv", lambda root: [])
    monkeypatch.setattr(launcher, "port_is_occupied", lambda port: True)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("spawned")))
    assert launcher.run(args(), root=tmp_path) == 2
    logs = list((tmp_path / "logs").glob("start-*.log"))
    assert len(logs) == 1
    assert "already in use" in logs[0].read_text()


def test_check_only_does_not_start_or_open_browser(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "check_venv", lambda root: [])
    monkeypatch.setattr(launcher, "port_is_occupied", lambda port: False)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("spawned")))
    monkeypatch.setattr(launcher.webbrowser, "open_new_tab", lambda *a: (_ for _ in ()).throw(AssertionError("opened")))
    assert launcher.run(args("--check-only"), root=tmp_path) == 0


class FakeProcess:
    def __init__(self, *a, **kw):
        self.stdout = StringIO("Booting Gradio\n")
        self.returncode = None
        self.wait_calls = []
    def poll(self):
        return self.returncode
    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        self.returncode = 0
        return 0
    def terminate(self):
        self.returncode = -15
    def kill(self):
        self.returncode = -9
    def send_signal(self, sig):
        self.returncode = -2


def test_ready_server_opens_browser_after_http_response(monkeypatch, tmp_path):
    fake_install(tmp_path)
    monkeypatch.setattr(launcher, "check_venv", lambda root: [])
    monkeypatch.setattr(launcher, "port_is_occupied", lambda port: False)
    monkeypatch.setattr(launcher.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(launcher, "http_is_ready", lambda port: True)
    opened = []
    monkeypatch.setattr(launcher.webbrowser, "open_new_tab", opened.append)
    assert launcher.run(args(), root=tmp_path) == 0
    assert opened == ["http://127.0.0.1:7860/"]
    logs = list((tmp_path / "logs").glob("start-*.log"))
    data = logs[0].read_text()
    assert "READY:" in data and "Booting Gradio" in data


def test_no_browser_honored(monkeypatch, tmp_path):
    fake_install(tmp_path)
    monkeypatch.setattr(launcher, "check_venv", lambda root: [])
    monkeypatch.setattr(launcher, "port_is_occupied", lambda port: False)
    monkeypatch.setattr(launcher.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(launcher, "http_is_ready", lambda port: True)
    monkeypatch.setattr(launcher.webbrowser, "open_new_tab", lambda u: (_ for _ in ()).throw(AssertionError("browser opened")))
    assert launcher.run(args("--no-browser"), root=tmp_path) == 0


def test_launcher_does_not_install_or_start_fastapi():
    source = SOURCE.read_text()
    assert "pip install" not in source
    assert "uvicorn" not in source.lower()
    assert "webbrowser.open_new_tab" in source

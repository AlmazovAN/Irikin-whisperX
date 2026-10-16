"""Offline tests of read-only Windows hardware/software diagnostic probes."""

import json
import subprocess
from types import SimpleNamespace
import urllib.error

import pytest

from modules.utils import environment_preflight as preflight
from scripts import diagnose_environment as cli

pytestmark = pytest.mark.fast


def test_probe_command_missing(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    assert preflight.probe_command("ffmpeg")["status"] == "missing"


def test_probe_command_version(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="ffmpeg version 7.0\nother\n"))
    assert preflight.probe_command("ffmpeg") == {"status": "ok", "version": "ffmpeg version 7.0"}


def test_probe_command_timeout(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/bin/tool")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="tool", timeout=1)
    monkeypatch.setattr(preflight.subprocess, "run", timeout)
    assert preflight.probe_command("ffmpeg")["status"] == "error"


def test_windows_gpu_adapters_flag_vram_as_unreliable(monkeypatch):
    monkeypatch.setattr(preflight.platform, "system", lambda: "Windows")
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "powershell.exe")
    win_json = json.dumps([{"Name": "AMD Radeon Graphics", "AdapterRAM": 512 * 1024**2, "DriverVersion": "30.1"}])
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=win_json))
    result = preflight.probe_windows_gpus()
    assert result["status"] == "ok"
    assert result["adapters"][0]["name"] == "AMD Radeon Graphics"
    assert result["adapters"][0]["vram_reliable"] is False
    assert "vram_mib" not in result["adapters"][0]


def test_windows_gpu_probe_without_powershell(monkeypatch):
    monkeypatch.setattr(preflight.platform, "system", lambda: "Windows")
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    assert preflight.probe_windows_gpus()["status"] == "unknown"


def test_nvidia_smi_real_vram(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="NVIDIA Test, 8192, 555.1\n"))
    result = preflight.probe_nvidia_smi()
    assert result["devices"][0]["vram_mib"] == 8192


def test_missing_packages_do_not_import_torch(monkeypatch):
    def no_import(name):
        pytest.fail("should not import missing torch")
    monkeypatch.setattr(preflight.importlib, "import_module", no_import)
    result = preflight.probe_torch({"torch": {"status": "missing"}})
    assert result["status"] == "missing"


def test_torch_cuda_failure_is_error_not_success(monkeypatch):
    def raise_import(name):
        raise RuntimeError("broken DLL")
    monkeypatch.setattr(preflight.importlib, "import_module", raise_import)
    result = preflight.probe_torch({"torch": {"status": "installed"}})
    assert result["status"] == "error"
    assert "RuntimeError" in result["detail"]


def test_torch_cpu_does_not_claim_cuda(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), version=SimpleNamespace(cuda=None))
    monkeypatch.setattr(preflight.importlib, "import_module", lambda name: fake_torch)
    result = preflight.probe_torch({"torch": {"status": "installed"}})
    assert result["cuda_available"] is False
    assert result["cuda_devices"] == []


def test_hf_presence_never_discloses_token(monkeypatch):
    secret = "hf_DO_NOT_PRINT_THIS_SECRET"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setattr(preflight.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("network must be opt in"))
    report = preflight.probe_hf()
    assert report["token_present"] is True
    assert report["authentication"] == "not_checked"
    assert secret not in json.dumps(report)


def test_hf_network_auth_only_on_opt_in(monkeypatch):
    secret = "hf_DO_NOT_PRINT_THIS_SECRET"
    monkeypatch.setenv("HF_TOKEN", secret)
    requests = []
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    def open_request(request, **kwargs):
        requests.append(request)
        return Response()
    monkeypatch.setattr(preflight.urllib.request, "urlopen", open_request)
    result = preflight.probe_hf(check_online=True)
    assert len(requests) == 1
    assert requests[0].full_url == preflight.HF_HOST
    assert result["authentication"] == "verified"
    assert result["model_access"] == "not_checked"
    assert secret not in json.dumps(result)


def test_hf_http_error_reports_only_status(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_sensitive")
    def denied(*a, **kw):
        raise urllib.error.HTTPError(preflight.HF_HOST, 401, "unauthorized", {}, None)
    monkeypatch.setattr(preflight.urllib.request, "urlopen", denied)
    assert preflight.probe_hf(check_online=True)["status"] == "auth_failed"


def test_preflight_blockers_and_warnings(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.platform, "system", lambda: "Windows")
    monkeypatch.setattr(preflight, "probe_windows_memory", lambda: {"status": "ok", "total_gib": 8, "available_gib": 3})
    monkeypatch.setattr(preflight, "probe_windows_gpus", lambda: {"status": "ok", "adapters": [{"name": "AMD Radeon Graphics"}]})
    monkeypatch.setattr(preflight, "probe_nvidia_smi", lambda: {"status": "not_available", "devices": []})
    monkeypatch.setattr(preflight, "probe_command", lambda cmd: {"status": "missing", "detail": "not found"} if cmd == "ffprobe" else {"status": "ok", "version": "ffmpeg test"})
    monkeypatch.setattr(preflight, "probe_packages", lambda: {name: {"status": "missing"} for name in preflight.PACKAGES})
    monkeypatch.setattr(preflight, "probe_hf", lambda **kw: {"status": "missing_token", "token_present": False})
    result = preflight.build_report(skip_torch=True, disk_path=tmp_path)
    assert result["assessment"]["inference_validated"] is False
    assert result["assessment"]["status"] == "blocked_prerequisites"
    assert any("ffprobe" in text for text in result["assessment"]["blockers"])
    assert any("12 GiB" in text for text in result["assessment"]["warnings"])
    assert "AMD Radeon" in preflight.render_text(result)


def test_cli_json_has_no_secrets(monkeypatch, capsys):
    mock_report = {"hello": "diagnostic"}
    monkeypatch.setattr(cli, "build_report", lambda **kwargs: mock_report)
    assert cli.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out) == mock_report


def test_cli_output_write_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "build_report", lambda **kwargs: {"test": 1})
    existing_file = tmp_path / "existing"
    existing_file.write_text("already", encoding="utf-8")
    assert cli.main(["--json", "--output", str(existing_file / "report.json")]) == 1

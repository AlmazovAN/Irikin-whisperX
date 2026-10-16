"""Pure Python security regression tests (no Gradio/FastAPI/torch imports)."""

import asyncio
import json

import pytest

from modules.utils.local_security import (
    LoopbackOnlyASGI,
    allowed_output_paths,
    is_local_host_header,
    is_local_origin,
    is_loopback_hostname,
    redact_config_secrets,
    validate_launch_settings,
)

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_launch_accepts_loopback_only(host):
    assert validate_launch_settings(server_name=host, share=False, api_open=False) == host


def test_launch_defaults_to_loopback():
    assert validate_launch_settings(server_name=None, share=False, api_open=False) == "127.0.0.1"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.0.1", "localhost.evil.example", "example.com", ""])
def test_launch_rejects_public_bind(host):
    with pytest.raises(ValueError, match="Only"):
        validate_launch_settings(server_name=host, share=False, api_open=False)


@pytest.mark.parametrize("flags", [{"share": True, "api_open": False}, {"share": False, "api_open": True}])
def test_launch_rejects_share_and_external_api(flags):
    with pytest.raises(ValueError, match="disabled"):
        validate_launch_settings(server_name="127.0.0.1", **flags)


@pytest.mark.parametrize("raw", ["__import__('os').system('echo bad')", "['/etc']", "{bad json}", '"subdir"', '{"foo": "bar"}'])
def test_allowed_paths_never_evaluates_python_or_accepts_non_json_lists(tmp_path, raw):
    with pytest.raises(ValueError):
        allowed_output_paths(str(tmp_path / "outputs"), raw)


def test_allowed_paths_default_only_output(tmp_path):
    out = (tmp_path / "output").resolve()
    assert allowed_output_paths(str(out), None) == [str(out)]


def test_allowed_paths_subdirectory(tmp_path):
    out = tmp_path / "output"
    child = out / "nested"
    assert allowed_output_paths(str(out), json.dumps([str(child)])) == [str(out.resolve()), str(child.resolve())]


@pytest.mark.parametrize("part", ["../secrets", "../../another", "."])
def test_allowed_paths_reject_traversal_or_root(tmp_path, part):
    out = tmp_path / "output"
    with pytest.raises(ValueError):
        allowed_output_paths(str(out), json.dumps([str(out / part)]))


def test_allowed_paths_rejects_symlink_escaping_output(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    external = tmp_path / "private"
    external.mkdir()
    link = out / "link"
    try:
        link.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable")
    with pytest.raises(ValueError):
        allowed_output_paths(str(out), json.dumps([str(link)]))


def test_secret_config_redaction_is_recursive_and_does_not_mutate():
    raw = {
        "whisper": {"suppress_tokens": [-1]},
        "diarization": {"hf_token": "hf_very_secret", "enable_offload": True},
        "translation": {"deepl": {"api_key": "deepl_secret"}},
        "nested": [{"refresh_token": "refresh_secret", "name": "keep"}],
    }
    result = redact_config_secrets(raw)
    assert raw["diarization"]["hf_token"] == "hf_very_secret"
    assert result["diarization"]["hf_token"] == ""
    assert result["translation"]["deepl"]["api_key"] == ""
    assert result["nested"][0] == {"refresh_token": "", "name": "keep"}
    assert result["whisper"]["suppress_tokens"] == [-1]
    assert "hf_very_secret" not in json.dumps(result)


@pytest.mark.parametrize("hostname,expected", [
    ("localhost", True), ("127.0.0.1", True), ("127.4.5.6", True), ("::1", True),
    ("0.0.0.0", False), ("8.8.8.8", False), ("localhost.evil.com", False),
    ("example.com", False),
])
def test_hostname_policy(hostname, expected):
    assert is_loopback_hostname(hostname) is expected


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1:7860", True), ("localhost", True), ("[::1]:7860", True),
    ("localhost.evil.com:7860", False), ("127.0.0.1@evil.com", False),
    ("0.0.0.0:7860", False), ("127.0.0.1:abc", False),
])
def test_host_header_policy(host, expected):
    assert is_local_host_header(host) is expected


@pytest.mark.parametrize("origin,expected", [
    (None, True), ("http://127.0.0.1:7860", True),
    ("http://localhost:7860", True), ("http://evil.com", False),
    ("null", False), ("https://localhost.evil.com", False),
    ("file:///", False),
])
def test_origin_policy(origin, expected):
    assert is_local_origin(origin) is expected


def run_asgi(*, scope_type="http", host="127.0.0.1:7860",
             client="127.0.0.1", origin=None, test_mode=False):
    seen = []
    headers = [(b"host", host.encode("latin-1"))]
    if origin is not None:
        headers.append((b"origin", origin.encode("latin-1")))

    async def downstream(scope, receive, send):
        seen.append("allowed")
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"OK"})
        else:
            await send({"type": "websocket.accept"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        seen.append(message)

    scope = {"type": scope_type, "client": (client, 1234), "headers": headers}
    asyncio.run(LoopbackOnlyASGI(downstream, test_mode=test_mode)(scope, receive, send))
    return seen


def test_local_http_request_allowed():
    assert "allowed" in run_asgi()


@pytest.mark.parametrize("params", [
    {"client": "192.168.1.2"},
    {"host": "evil.example"},
    {"origin": "http://evil.example"},
    {"origin": "http://127.0.0.1:9999"},
    {"host": "localhost.evil.example"},
])
def test_remote_and_cross_port_http_denied(params):
    msgs = run_asgi(**params)
    assert "allowed" not in msgs
    assert msgs[0]["status"] == 403


def test_external_websocket_denied():
    result = run_asgi(scope_type="websocket", client="192.168.1.44")
    assert "allowed" not in result
    assert result[0] == {"type": "websocket.close", "code": 1008}


def test_testclient_bypass_only_in_explicit_test_mode():
    assert "allowed" not in run_asgi(client="testclient", host="testserver")
    assert "allowed" in run_asgi(client="testclient", host="testserver", test_mode=True)

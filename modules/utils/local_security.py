"""Security boundaries for a single-user, loopback-only WhisperX instance.

Pure standard-library module so rules can be exercised by fast offline tests.
Not a substitute for OS firewall permissions or a multi-user authentication layer.
"""

from __future__ import annotations

from copy import deepcopy
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit

SECRET_KEYS = frozenset({
    "hf_token", "api_key", "auth_key", "access_token", "refresh_token",
    "client_secret", "password",
})


def redact_config_secrets(value):
    """Copy a YAML-friendly structure without retaining authentication secrets."""
    if isinstance(value, dict):
        return {
            key: ("" if str(key).lower() in SECRET_KEYS else redact_config_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_config_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_config_secrets(item) for item in value)
    return deepcopy(value)


def is_loopback_hostname(host: str | None) -> bool:
    if not isinstance(host, str) or not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_local_host_header(host_header: str | None) -> bool:
    if not host_header:
        return False
    try:
        parsed = urlsplit("//" + host_header)
        port = parsed.port
        if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            return False
        return is_loopback_hostname(parsed.hostname) and (port is None or 1 <= port <= 65535)
    except ValueError:
        return False


def is_local_origin(origin: str | None) -> bool:
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
        return (
            parsed.scheme in ("http", "https")
            and not parsed.username and not parsed.password
            and is_loopback_hostname(parsed.hostname)
            and (parsed.port is None or 1 <= parsed.port <= 65535)
            and not parsed.path and not parsed.query and not parsed.fragment
        )
    except ValueError:
        return False


def validate_launch_settings(*, server_name: str | None, share: bool,
                             api_open: bool) -> str:
    """Fail closed instead of silently permitting an externally reachable UI."""
    host = server_name or "127.0.0.1"
    if host not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Only 127.0.0.1, localhost and ::1 are permitted for local mode.")
    if share:
        raise ValueError("Public Gradio sharing (--share) is disabled in local mode.")
    if api_open:
        raise ValueError("Public Gradio queue API (--api_open) is disabled in local mode.")
    return host


def allowed_output_paths(output_dir: str, extra_json: str | None) -> list[str]:
    """Permit only output directory and explicitly named children, never parent paths."""
    root = Path(output_dir).expanduser().resolve()
    if root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError("output_dir must not be a drive root or home directory.")
    paths = [str(root)]
    if not extra_json:
        return paths
    try:
        parsed = json.loads(extra_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("--allowed_paths requires a JSON list of output subdirectories.") from exc
    if not isinstance(parsed, list) or any(not isinstance(x, str) or not x for x in parsed):
        raise ValueError("--allowed_paths must be a JSON list of non-empty paths.")
    for raw in parsed:
        path = Path(raw).expanduser().resolve()
        if path == root or root not in path.parents:
            raise ValueError("--allowed_paths may only name subdirectories of output_dir.")
        if path.is_file():
            raise ValueError("--allowed_paths entries must be directories.")
        if str(path) not in paths:
            paths.append(str(path))
    return paths


class LoopbackOnlyASGI:
    """Reject non-loopback HTTP/WebSocket traffic, Host spoofing and foreign Origins.

    TestClient's synthetic client/host pair can be allowed ONLY when TEST_ENV is
    explicitly set; never enable TEST_ENV when running a real server.
    """

    def __init__(self, app, *, test_mode: bool = False):
        self.app = app
        self.test_mode = test_mode

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): val.decode("latin-1") for key, val in scope.get("headers", [])}
        client = scope.get("client")
        client_host = client[0] if client else None
        host = headers.get(b"host")
        origin = headers.get(b"origin")
        synthetic_test_client = (
            self.test_mode and client_host == "testclient" and host == "testserver"
        )
        allowed = synthetic_test_client or (
            is_loopback_hostname(client_host)
            and is_local_host_header(host)
            and is_local_origin(origin)
            and (not origin or urlsplit(origin).netloc.lower() == host.lower())
        )
        if allowed:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            await send({
                "type": "http.response.start", "status": 403,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")],
            })
            await send({"type": "http.response.body", "body": b"Localhost access only"})

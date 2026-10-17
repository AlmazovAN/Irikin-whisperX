"""PR21 / P20: Simple UI offline tests, real SQLite/files, fake Gradio/FFprobe.

No models, browsers, Gradio package, external network or secrets required.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import ast
import subprocess
import sys
import wave

import pytest

from modules.jobs.service import JobService
from modules.ui import simple_mode
from modules.ui.simple_mode import SimpleModeController, build_blocks, upload_directory

pytestmark = pytest.mark.fast
ROOT = Path(__file__).resolve().parents[2]


class FakeWorker:
    def __init__(self, jobs):
        self.jobs = jobs
        self.is_running = False
        self.wakes = 0
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1
        self.is_running = True

    def wake(self):
        self.wakes += 1

    def stop(self, *, wait=True):
        assert wait is True
        self.stops += 1
        self.is_running = False
        return True


@pytest.fixture(autouse=True)
def stub_ffprobe(monkeypatch):
    monkeypatch.setattr(
        "modules.jobs.inputs._probe_audio",
        lambda path, *, binary, timeout: (1.0, "pcm_s16le"),
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    stage = tmp_path / "data" / "ui_uploads"
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(stage))
    root = tmp_path / "data" / "jobs"
    jobs = JobService(storage_root=root, upload_roots=(upload_directory(),))
    worker = FakeWorker(jobs)
    worker.start()
    return jobs, worker, stage


def wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\0\0" * 4000)
    return path


def test_simple_controller_accepts_durable_upload_with_two_actions(env):
    jobs, worker, staging = env
    supplied = wav(staging / "hash" / "sample.wav")
    controller = SimpleModeController(jobs, worker)
    identifier, status = controller.submit(str(supplied))
    assert "QUEUED" in status
    assert len(identifier) == 36
    assert worker.wakes == 1
    ticket = jobs.get(identifier)
    assert ticket.status == "QUEUED"
    assert ticket.model_name == "small" and ticket.language == "ru"
    assert jobs.saved_input(identifier).is_file()
    assert jobs.saved_input(identifier).parent.parent.parent == jobs.storage.root
    assert supplied.is_file()  # Does not destructively manipulate staged file.


def test_named_filepath_upload_is_supported(env):
    jobs, worker, stage = env
    supplied = wav(stage / "hash" / "record.m4a")
    accepted, status = SimpleModeController(jobs, worker).submit(
        SimpleNamespace(name=str(supplied))
    )
    assert accepted and status.startswith("В очереди")
    assert jobs.get(accepted).file_name == "record.m4a"


def test_no_audio_is_friendly_and_not_queued(env):
    jobs, worker, _ = env
    controller = SimpleModeController(jobs, worker)
    assert controller.submit(None) == ("", "Выберите аудиофайл.")
    assert controller.submit("") == ("", "Выберите аудиофайл.")
    assert controller.submit([])[0] == ""
    assert jobs.list_jobs() == ()
    assert worker.wakes == 0


def test_file_outside_gradio_upload_root_is_denied(env, tmp_path):
    jobs, worker, _ = env
    outside = wav(tmp_path / "other" / "secret.wav")
    uuid, feedback = SimpleModeController(jobs, worker).submit(str(outside))
    assert uuid == ""
    assert "ACCESS_DENIED" in feedback
    assert str(outside) not in feedback
    assert jobs.list_jobs() == ()
    assert worker.wakes == 0


def test_symlink_escape_is_not_authorized(env, tmp_path):
    jobs, worker, stage = env
    outside = wav(tmp_path / "secret" / "private.wav")
    link = stage / "escaped.wav"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Platform does not permit symlink creation")
    uuid, feedback = SimpleModeController(jobs, worker).submit(str(link))
    assert uuid == "" and "ACCESS_DENIED" in feedback
    assert jobs.list_jobs() == ()


def test_not_running_worker_cannot_accept_orphan_queue(env):
    jobs, worker, stage = env
    worker.is_running = False
    supplied = wav(stage / "hash" / "not_accepted.wav")
    uid, feedback = SimpleModeController(jobs, worker).submit(str(supplied))
    assert uid == ""
    assert "WORKER_UNAVAILABLE" in feedback
    assert jobs.list_jobs() == ()


def test_controller_refuses_two_unrelated_queue_owners(env):
    jobs, worker, stage = env
    other = JobService(storage_root=stage / "other_jobs")
    with pytest.raises(ValueError, match="same|worker"):
        SimpleModeController(other, worker)


def test_upload_root_created_and_symlinked_root_refused(tmp_path, monkeypatch):
    directory = tmp_path / "upload"
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(directory))
    assert upload_directory() == directory.resolve()
    assert directory.is_dir()
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    directory.rmdir()
    try:
        directory.symlink_to(destination, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable")
    with pytest.raises(Exception) as exc:
        upload_directory()
    assert getattr(exc.value, "code", None) == "UNSAFE_UPLOAD_ROOT"


def test_create_runtime_uses_same_service_and_trusted_temp_root(tmp_path, monkeypatch):
    staging = tmp_path / "ui_cache"
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(staging))
    controller, worker = simple_mode.create_runtime(job_root=tmp_path / "jobs")
    assert controller.worker is worker
    assert controller.jobs is worker.jobs
    assert controller.jobs.upload_roots == (staging.resolve(),)
    assert not worker.is_running


class FakeContext:
    def __init__(self, owner=None):
        self.owner = owner

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def load(self, **kwargs):
        self.owner.loads.append(kwargs)


class FakeDependency:
    def __init__(self, owner):
        self.owner = owner

    def then(self, **kwargs):
        self.owner.thens.append(kwargs)
        return self


class FakeElement:
    def __init__(self, owner, name, **kwargs):
        self.owner = owner
        self.name = name
        self.kwargs = kwargs

    def click(self, **kwargs):
        self.owner.clicks.append(kwargs)
        return FakeDependency(self.owner)

    def select(self, **kwargs):
        self.owner.selects.append(kwargs)

    def tick(self, **kwargs):
        self.owner.ticks.append(kwargs)

    def change(self, **kwargs):
        self.owner.changes.append(kwargs)


class FakeGradio:
    def __init__(self):
        self.markdowns = []
        self.elements = []
        self.clicks = []
        self.tabs = []
        self.loads = []
        self.ticks = []
        self.changes = []
        self.selects = []
        self.thens = []

    @staticmethod
    def update(**kwargs):
        return kwargs

    def Blocks(self):
        return FakeContext(self)

    def Tabs(self):
        return FakeContext()

    def Tab(self, name):
        self.tabs.append(name)
        return FakeContext()

    def Markdown(self, text):
        self.markdowns.append(text)

    def File(self, **kwargs):
        obj = FakeElement(self, "File", **kwargs)
        self.elements.append(obj)
        return obj

    def Button(self, title, **kwargs):
        obj = FakeElement(self, "Button", title=title, **kwargs)
        self.elements.append(obj)
        return obj

    def Textbox(self, **kwargs):
        obj = FakeElement(self, "Textbox", **kwargs)
        self.elements.append(obj)
        return obj

    def Slider(self, **kwargs):
        obj = FakeElement(self, "Slider", **kwargs)
        self.elements.append(obj)
        return obj

    def BrowserState(self, value, **kwargs):
        obj = FakeElement(self, "BrowserState", value=value, **kwargs)
        self.elements.append(obj)
        return obj

    def Timer(self, **kwargs):
        obj = FakeElement(self, "Timer", **kwargs)
        self.elements.append(obj)
        return obj

    def DownloadButton(self, title, **kwargs):
        obj = FakeElement(self, "DownloadButton", title=title, **kwargs)
        self.elements.append(obj)
        return obj

    def Dropdown(self, **kwargs):
        obj = FakeElement(self, "Dropdown", **kwargs)
        self.elements.append(obj)
        return obj

    def Checkbox(self, **kwargs):
        obj = FakeElement(self, "Checkbox", **kwargs)
        self.elements.append(obj)
        return obj


def test_simple_blocks_have_drag_drop_one_launch_button_and_separate_advanced(env):
    jobs, worker, stage = env
    gr = FakeGradio()
    blocks = build_blocks(SimpleModeController(jobs, worker), gradio=gr)
    assert isinstance(blocks, FakeContext)
    assert gr.tabs == ["Simple", "Advanced"]
    assert len([e for e in gr.elements if e.name == "File"]) == 1
    uploads = next(e for e in gr.elements if e.name == "File")
    assert uploads.kwargs["type"] == "filepath"
    assert uploads.kwargs["file_count"] == "single"
    assert sorted(uploads.kwargs["file_types"]) == [".m4a", ".mp3", ".ogg", ".wav"]
    buttons = [e for e in gr.elements if e.name == "Button"]
    assert any(b.kwargs["title"] == "Начать обработку" for b in buttons)
    assert any(b.kwargs["title"] == "Обновить историю" for b in buttons)
    assert any(b.kwargs["title"] == "Открыть папку результата" for b in buttons)
    assert any(b.kwargs["title"] == "Удалить выбранное задание" for b in buttons)
    callback = gr.clicks[0]
    assert callback["api_name"] is False
    browser = next(e for e in gr.elements if e.name == "BrowserState")
    slider = next(e for e in gr.elements if e.name == "Slider")
    assert callback["inputs"] == [uploads, browser]
    assert len(callback["outputs"]) == 7
    assert callback["outputs"][-3] is browser
    download = next(e for e in gr.elements if e.name == "DownloadButton")
    assert download.kwargs["title"] == "Скачать TXT"
    assert download.kwargs["value"] is None
    assert callback["outputs"][-2] is download
    assert callback["outputs"][-1].name == "Textbox"
    assert len(gr.changes) == 1
    assert gr.changes[0]["inputs"] == [browser]
    assert gr.changes[0]["outputs"] == []
    assert gr.changes[0]["fn"] is None
    assert "localStorage" in gr.changes[0]["js"]
    assert "click()" in gr.changes[0]["js"]
    assert slider.kwargs["interactive"] is False
    assert browser.kwargs["storage_key"] and browser.kwargs["secret"]
    assert len(gr.loads) == 2 and len(gr.ticks) == 1
    assert len(gr.selects) == 1 and len(gr.thens) >= 3
    assert len([e for e in gr.elements if e.name == "Dropdown"]) == 1
    assert len([e for e in gr.elements if e.name == "Checkbox"]) == 1
    assert gr.loads[0]["inputs"] == [browser]
    assert gr.ticks[0]["inputs"] == [browser]
    assert gr.loads[0]["queue"] is False
    assert gr.ticks[0]["queue"] is False
    assert gr.ticks[0]["outputs"] == gr.loads[0]["outputs"]
    assert gr.ticks[0]["outputs"][-2] is download
    assert any("Small" in s and "ru" in s for s in gr.markdowns)
    assert any("Start.bat --ui-mode advanced" in s for s in gr.markdowns)

    # The actual Gradio-bound function performs only a durable submit.
    uid, response, stage_text, milestone, saved, file, preview = callback["fn"](
        str(wav(stage / "hash" / "ui.wav")), ""
    )
    assert jobs.get(uid).status == "QUEUED"
    assert response.startswith("В очереди")
    assert saved == uid and stage_text == "В очереди" and milestone == 0
    assert file is None and preview == ""
    assert gr.loads[0]["fn"](saved)[0] == uid
    history_choice = gr.loads[1]["fn"](saved)
    assert history_choice["value"] == uid
    assert any(u == uid for _, u in history_choice["choices"])
    assert gr.ticks[0]["fn"](saved)[0] == uid
    assert worker.wakes == 1


def test_empty_ui_event_does_not_initialize_whisper_model(env):
    jobs, worker, stage = env
    gr = FakeGradio()
    build_blocks(SimpleModeController(jobs, worker), gradio=gr)
    assert jobs.list_jobs() == ()
    assert worker.wakes == 0
    gr.clicks[0]["fn"](None, "")
    assert jobs.list_jobs() == ()


def test_simple_module_import_has_no_heavy_ml_or_gradio_side_effects():
    script = (
        "import modules.ui.simple_mode; import sys; "
        "assert 'gradio' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'whisperx' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        timeout=15, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_app_switch_defaults_to_simple_before_legacy_app_initialization():
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    controls = [node for node in ast.walk(tree)
                if isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and getattr(node.test.left, "attr", None) == "ui_mode"]
    assert controls
    branch = controls[-1]
    calls = [n for n in ast.walk(branch) if isinstance(n, ast.Call)]
    invoked = [getattr(n.func, "id", None) for n in calls]
    assert "launch_simple" in invoked
    assert any(getattr(n.func, "id", None) == "App" for n in calls)
    assert (ROOT / "app.py").read_text(encoding="utf-8").index(
        'os.environ.setdefault('
    ) < (ROOT / "app.py").read_text(encoding="utf-8").index("import gradio as gr")


def test_windows_launcher_supports_simple_default_and_exclusive_advanced():
    import importlib.util
    path = ROOT / "scripts" / "launch_windows.py"
    spec = importlib.util.spec_from_file_location("pr21_windows_launcher", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    default = mod.parse_args([])
    advanced = mod.parse_args(["--ui-mode", "advanced"])
    assert default.ui_mode == "simple"
    assert advanced.ui_mode == "advanced"
    default_cmd = mod.build_app_command(Path("C:/repo/python.exe"), default)
    advanced_cmd = mod.build_app_command(Path("C:/repo/python.exe"), advanced)
    assert default_cmd[default_cmd.index("--ui_mode") + 1] == "simple"
    assert advanced_cmd[advanced_cmd.index("--ui_mode") + 1] == "advanced"
    with pytest.raises(SystemExit):
        mod.parse_args(["--ui-mode", "unexpected"])



def _launch_args(**overrides):
    kwargs = dict(
        server_name="127.0.0.1", share=False, api_open=False,
        server_port=7860, username=None, password=None, root_path=None,
        inbrowser=False, ssl_verify=True, ssl_keyfile=None,
        ssl_keyfile_password=None, ssl_certfile=None,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def test_server_starts_worker_before_browser_accepts_requests_and_stops(monkeypatch, env):
    jobs, worker, _ = env
    worker.is_running = False
    history = []

    class FakeBlocks:
        def queue(self, **kwargs):
            assert kwargs["api_open"] is False
            history.append("queue")
            return self

        def launch(self, **kwargs):
            assert kwargs["server_name"] == "127.0.0.1"
            assert kwargs["share"] is False
            assert worker.is_running
            history.append("launch")

    monkeypatch.setattr(simple_mode, "create_runtime",
                        lambda: (SimpleModeController(jobs, worker), worker))
    monkeypatch.setattr(simple_mode, "build_blocks",
                        lambda controller, **kwargs: FakeBlocks())
    simple_mode.launch_simple(_launch_args())
    assert worker.starts == 2  # The env fixture first started this fake worker.
    assert worker.stops == 1
    assert history == ["queue", "launch"]
    assert not worker.is_running


def test_server_launch_failure_releases_worker(monkeypatch, env):
    jobs, worker, _ = env
    worker.is_running = False

    class FailingBlocks:
        def queue(self, **kwargs):
            return self

        def launch(self, **kwargs):
            assert worker.is_running
            raise RuntimeError("startup failure")

    monkeypatch.setattr(simple_mode, "create_runtime",
                        lambda: (SimpleModeController(jobs, worker), worker))
    monkeypatch.setattr(simple_mode, "build_blocks",
                        lambda controller, **kwargs: FailingBlocks())
    with pytest.raises(RuntimeError, match="startup failure"):
        simple_mode.launch_simple(_launch_args())
    assert worker.stops == 1
    assert not worker.is_running


def test_remote_host_refused_before_worker_start(monkeypatch, env):
    jobs, worker, _ = env
    before = worker.starts
    monkeypatch.setattr(simple_mode, "create_runtime",
                        lambda: (SimpleModeController(jobs, worker), worker))
    with pytest.raises(ValueError, match="127.0.0.1"):
        simple_mode.launch_simple(_launch_args(server_name="0.0.0.0"))
    assert worker.starts == before

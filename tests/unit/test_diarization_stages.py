"""PR12 offline checks of the actual alignment/Pyannote adapter methods.

Extract production methods with AST: CI can execute their logic without
loading torch, pyannote, whisperx, pandas, Gradio or model downloads.
A synthetic two-speaker fixture is NOT real-model acoustic validation.
"""

import ast
import hashlib
import inspect
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.diarize.diagnostics import (
    HF_ACCESS_GUIDANCE, ModelStageError, require_device, require_hf_token,
)

pytestmark = pytest.mark.fast
ROOT = Path(__file__).resolve().parents[2]


def load_method(path, classname, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    parent = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == classname)
    method = next(x for x in parent.body if isinstance(x, ast.FunctionDef) and x.name == name)
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("value", [None, "", "  ", "\t"])
def test_gated_token_missing_has_clear_error(value):
    with pytest.raises(ModelStageError) as e:
        require_hf_token(value)
    assert e.value.code == "HF_TOKEN_MISSING"
    assert "HF_TOKEN" in str(e.value)
    assert "private" in str(e.value)


@pytest.mark.parametrize("device,cuda,code", [
    ("cuda", False, "CUDA_UNAVAILABLE"),
    ("xpu", False, "DEVICE_UNSUPPORTED"),
])
def test_cuda_availability_is_explicit(device, cuda, code):
    with pytest.raises(ModelStageError) as e:
        require_device(device, cuda)
    assert e.value.code == code


def test_cpu_works_without_cuda_and_cuda_when_available():
    require_device("cpu", False)
    require_device("cuda", True)


def pyannote_init(pipe_factory, *, cuda_available=False):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        device=lambda name: name,
    )
    namespace = dict(
        os=os, torch=fake_torch, Pipeline=SimpleNamespace(from_pretrained=pipe_factory),
        WHISPERX_MODELS_DIR="/tmp/fake-no-write",
        require_device=require_device, require_hf_token=require_hf_token,
        ModelStageError=ModelStageError, HF_ACCESS_GUIDANCE=HF_ACCESS_GUIDANCE,
    )
    init = load_method("modules/diarize/diarize_pipeline.py", "DiarizationPipeline",
                       "__init__", namespace)
    instance = SimpleNamespace()
    return init, instance


def test_missing_token_never_calls_remote_pyannote_factory():
    touched = []
    init, obj = pyannote_init(lambda *a, **k: touched.append(True))
    with pytest.raises(ModelStageError) as e:
        init(obj, cache_dir=".", use_auth_token=None)
    assert e.value.code == "HF_TOKEN_MISSING"
    assert touched == []


def test_gated_pyannote_returns_none_without_attributeerror():
    init, obj = pyannote_init(lambda *a, **k: None)
    with pytest.raises(ModelStageError) as e:
        init(obj, cache_dir=".", use_auth_token="hf_fake_unit_test")
    assert e.value.code == "HF_MODEL_ACCESS"
    assert "pyannote" in str(e.value).lower()


def test_pyannote_denied_download_does_not_print_secret_or_library_error():
    secret = "hf_test_super_private_not_real"
    def forbidden(*args, **kwargs):
        raise RuntimeError("private token "+secret)
    init, obj = pyannote_init(forbidden)
    with pytest.raises(ModelStageError) as e:
        init(obj, cache_dir=".", use_auth_token=secret)
    assert e.value.code == "HF_MODEL_ACCESS"
    assert secret not in str(e.value)


def test_pyannote_cpu_and_cuda_selection_with_authorized_stub():
    device_seen = []
    class FakeModel:
        def to(self, *, device):
            device_seen.append(device)
            return self
    model = FakeModel()
    for choice, cuda in (("cpu", False), ("cuda", True)):
        init, obj = pyannote_init(lambda *a, **kw: model, cuda_available=cuda)
        init(obj, cache_dir=".", use_auth_token="hf_fake_unit_test", device=choice)
        assert obj.model is model
    assert device_seen == ["cpu", "cuda"]


def test_pyannote_explicit_cuda_on_cpu_fails_before_model_load():
    touched = []
    init, obj = pyannote_init(lambda *args, **kw: touched.append(True))
    with pytest.raises(ModelStageError) as e:
        init(obj, cache_dir=".", use_auth_token="hf_fake_unit_test", device="cuda")
    assert e.value.code == "CUDA_UNAVAILABLE"
    assert not touched


def test_pyannote_device_oom_message_is_safe():
    class Model:
        def to(self, *, device):
            raise RuntimeError("secret token=hf_do_not_echo: OOM")
    init, obj = pyannote_init(lambda *a, **k: Model())
    with pytest.raises(ModelStageError) as e:
        init(obj, cache_dir=".", use_auth_token="hf_fake_unit_test")
    assert e.value.code == "DIARIZATION_MODEL_LOAD"
    assert "hf_do_not_echo" not in str(e.value)


def wrapper_method(name, *, whisperx=None, diarizer=None, assign=None,
                   cuda_available=False, audio_loader=None):
    namespace = dict(
        inspect=inspect, hashlib=hashlib, os=os, logger=logging.getLogger("test-pr12"),
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available)),
        require_device=require_device, require_hf_token=require_hf_token,
        ModelStageError=ModelStageError,
        whisperx=whisperx, DiarizationPipeline=diarizer,
        assign_word_speakers=assign, load_audio=audio_loader or (lambda a:a),
    )
    method = load_method("modules/whisper/whisperx_wrapper.py", "WhisperXWrapper",
                         name, namespace)
    return method


def test_alignment_uses_russian_and_cpu_and_keeps_word_timestamps():
    called = {}
    def fake_load_align_model(*, language_code, device, model_dir):
        called["load"] = (language_code, device, model_dir)
        return object(), {"vocab": "russian"}
    def fake_align(segments, model, metadata, audio, device, *, return_char_alignments):
        called["aligned"] = (audio, device, return_char_alignments)
        return {"segments": [{
            "start": 0.1, "end": 0.9, "text": "Привет",
            "words": [{"start": 0.15, "end": 0.85, "word": "Привет"}],
        }]}
    fake_whisperx = SimpleNamespace(
        load_align_model=fake_load_align_model, align=fake_align,
    )
    align = wrapper_method("align", whisperx=fake_whisperx)
    owner = SimpleNamespace(
        _require_whisperx=lambda: None, _resolve_device=lambda d:d,
        _align_model=None, _align_metadata=None, _align_language=None,
        model_dir="models/Whisper",
        _segments_to_dict=lambda segs: segs,
        _segments_from_dict=lambda segs: segs,
    )
    result = align(owner, [{"text": "Привет"}], "ru", "original-audio", "cpu")
    assert called["load"] == ("ru", "cpu", "models/Whisper")
    assert called["aligned"] == ("original-audio", "cpu", False)
    assert result[0]["words"][0]["start"] == 0.15


def test_alignment_load_failure_is_sanitized():
    def bad_align_loader(**kwargs):
        raise RuntimeError("private local path and hf_secret")
    align = wrapper_method(
        "align", whisperx=SimpleNamespace(load_align_model=bad_align_loader))
    owner = SimpleNamespace(
        _require_whisperx=lambda: None, _resolve_device=lambda d:d,
        _align_model=None, _align_metadata=None, _align_language=None,
        model_dir="models/Whisper")
    with pytest.raises(ModelStageError) as e:
        align(owner, [object()], "ru", "original-audio", "cpu")
    assert e.value.code == "ALIGNMENT_MODEL_UNAVAILABLE"
    assert "hf_secret" not in str(e.value)


def test_alignment_runtime_error_has_separate_code():
    def bad_align(*args, **kwargs):
        raise RuntimeError("private token")
    align = wrapper_method("align", whisperx=SimpleNamespace(
        load_align_model=lambda **kwargs:(object(), object()), align=bad_align))
    owner = SimpleNamespace(
        _require_whisperx=lambda: None, _resolve_device=lambda d:d,
        _align_model=None, _align_metadata=None, _align_language=None,
        model_dir="models/Whisper", _segments_to_dict=lambda x:x)
    with pytest.raises(ModelStageError) as e:
        align(owner, [object()], "ru", "original-audio", "cpu")
    assert e.value.code == "ALIGNMENT_FAILED"
    assert "private" not in str(e.value)


def test_whisperx_diarize_assigns_two_speakers_to_words():
    called = {"factory": 0, "assignment": []}
    secret = "hf_test_private_fake"
    def create_diarizer(**kwargs):
        called["factory"] += 1
        assert kwargs["device"] == "cpu"
        assert kwargs["compute_type"] == "float32"
        assert kwargs["use_auth_token"] == secret
        return lambda audio: "two-speaker-timeline"
    def fake_assign(diarized, transcribed, **kwargs):
        called["assignment"].append((diarized, kwargs))
        assert transcribed["segments"] == [{"text": "две реплики"}]
        return {"segments": [
            {"text": "Добрый день", "speaker": "SPEAKER_00",
             "words": [{"word": "Добрый", "speaker": "SPEAKER_00", "start": 0, "end": 0.4}]},
            {"text": "Здравствуйте", "speaker": "SPEAKER_01",
             "words": [{"word": "Здравствуйте", "speaker": "SPEAKER_01", "start": 1, "end": 1.5}]},
        ]}
    diarize = wrapper_method("diarize", diarizer=create_diarizer, assign=fake_assign)
    owner = SimpleNamespace(
        _resolve_device=lambda d:d,
        _available_diarization_compute_types=lambda d:["float32"],
        _segments_to_dict=lambda s:s,
        _segments_from_dict=lambda s, **kw:s,
        diarization_model_dir="models/Diarization",
        _diarization_pipeline=None, _diarization_device=None,
        _diarization_token_fingerprint=None, _diarization_compute_type=None,
    )
    params = SimpleNamespace(diarization_device="cpu",hf_token=secret,
                             compute_type="float32",fill_nearest_speaker=False,
                             assign_word_speakers=True)
    result = diarize(owner, "original-audio", [{"text": "две реплики"}], params)
    assert called["factory"] == 1
    assert [s["speaker"] for s in result] == ["SPEAKER_00", "SPEAKER_01"]
    assert [s["words"][0]["speaker"] for s in result] == ["SPEAKER_00", "SPEAKER_01"]
    assert called["assignment"] == [
        ("two-speaker-timeline", {"fill_nearest": False, "tag_words": True})]
    assert owner._diarization_token_fingerprint == hashlib.sha256(secret.encode()).digest()
    assert secret.encode() != owner._diarization_token_fingerprint
    diarize(owner, "original-audio", [{"text": "две реплики"}], params)
    assert called["factory"] == 1


def test_whisperx_diarize_missing_token_precedes_model_loading(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    touched = []
    diarize = wrapper_method(
        "diarize", diarizer=lambda **kw: touched.append(True),
        assign=lambda *a, **kw:{})
    owner = SimpleNamespace(
        _resolve_device=lambda d:d,
        _available_diarization_compute_types=lambda d:["float32"],
        _diarization_pipeline=None, _diarization_device=None,
        _diarization_token_fingerprint=None, _diarization_compute_type=None)
    params = SimpleNamespace(diarization_device="cpu", hf_token="", compute_type="float32")
    with pytest.raises(ModelStageError) as e:
        diarize(owner, "audio", [{"text": "hello"}], params)
    assert e.value.code == "HF_TOKEN_MISSING"
    assert touched == []

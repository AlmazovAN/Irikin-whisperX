"""Offline P06 smoke harness tests. Never import Gradio, torch or actual WhisperX."""

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "smoke_whisperx.py"
spec = importlib.util.spec_from_file_location("p06_smoke", SOURCE)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

pytestmark = pytest.mark.fast


def sample_report(*, hf=True, cuda=False, os_name="Windows", missing=(), blockers=()):
    packages = {
        name: {"status": "installed", "version": "test-version"}
        for name in smoke.REQUIRED_PACKAGES
    }
    for name in missing:
        packages[name] = {"status": "missing"}
    return {
        "system": {"name": os_name, "release": "11", "machine": "AMD64"},
        "python": {"version": "3.11.9", "supported_by_readme": True},
        "ram": {"status": "ok", "total_gib": 8},
        "disk": {"status": "ok", "free_gib": 21},
        "ffmpeg": {"status": "ok"},
        "ffprobe": {"status": "ok"},
        "windows_gpus": {"adapters": [{"name": "AMD Radeon Graphics"}]},
        "nvidia_smi": {"devices": []},
        "torch": {"status": "ok", "cuda_available": cuda, "torch_cuda_build": None},
        "packages": packages,
        "hugging_face": {"token_present": hf, "status": "token_present"},
        "assessment": {"blockers": list(blockers), "warnings": ["CPU is slower"]},
    }


def test_cli_requires_either_diagnosis_or_audio():
    with pytest.raises(SystemExit):
        smoke.parse_args([])


@pytest.mark.parametrize("values", [
    ["--audio", "x.wav", "--diagnose-only"],
    ["--diagnose-only", "--batch-size", "0"],
    ["--diagnose-only", "--batch-size", "9"],
    ["--diagnose-only", "--device", "cpu", "--compute-type", "float16"],
    ["--diagnose-only", "--device", "metal"],
])
def test_cli_refuses_invalid_config(values):
    with pytest.raises(SystemExit):
        smoke.parse_args(values)


def test_cpu_small_defaults_and_local_audio_path():
    args = smoke.parse_args(["--audio", "local-file.wav"])
    assert args.device == "cpu"
    assert args.compute_type == "int8"
    assert args.batch_size == 1
    assert args.audio == Path("local-file.wav")
    assert not args.without_diarization


def test_no_diarization_still_explicit_opt_out():
    args = smoke.parse_args(["--diagnose-only", "--without-diarization"])
    assert args.without_diarization


def test_full_pipeline_blocked_without_hf_token():
    blockers = smoke.prerequisites(sample_report(hf=False), device="cpu", require_diarization=True)
    assert any("HF_TOKEN" in item for item in blockers)


def test_asr_only_not_blocked_for_missing_hf_token():
    blockers = smoke.prerequisites(sample_report(hf=False), device="cpu", require_diarization=False)
    assert not any("HF_TOKEN" in item for item in blockers)


def test_cuda_not_inferred_from_amd():
    blockers = smoke.prerequisites(sample_report(cuda=False), device="cuda", require_diarization=True)
    assert any("CUDA" in item for item in blockers)


def test_missing_package_is_blocker():
    result = smoke.prerequisites(sample_report(missing=("whisperx",)), device="cpu", require_diarization=True)
    assert any("whisperx" in item for item in result)


def test_linux_never_claims_windows_acceptance():
    blockers = smoke.prerequisites(sample_report(os_name="Linux"), device="cpu", require_diarization=True)
    assert any("Windows" in item for item in blockers)


def test_preflight_blocker_preserved():
    problems = smoke.prerequisites(sample_report(blockers=("FFmpeg unavailable",)), device="cpu", require_diarization=True)
    assert "FFmpeg unavailable" in problems


def test_duplicate_blocker_removed():
    problems = smoke.prerequisites(
        sample_report(blockers=("PyTorch device probe has not succeeded",)),
        device="cpu", require_diarization=True,
    )
    assert len(problems) == len(set(problems))


def test_missing_file_blocked_without_audio_loading(tmp_path):
    with pytest.raises(smoke.SmokeBlocked, match="does not exist"):
        smoke.inspect_audio(tmp_path / "no-such-file.wav")


@pytest.mark.parametrize("ext", [".txt", ".m4a", ".exe"])
def test_unsupported_audio_format_blocked(tmp_path, ext):
    path = tmp_path / ("audio" + ext)
    path.write_bytes(b"not audio")
    with pytest.raises(smoke.SmokeBlocked, match="WAV and MP3"):
        smoke.inspect_audio(path)


def test_zero_and_oversize_files_blocked(tmp_path):
    path = tmp_path / "zero.wav"
    path.write_bytes(b"")
    with pytest.raises(smoke.SmokeBlocked, match="25 MiB"):
        smoke.inspect_audio(path)
    path.write_bytes(b"x" * 100)
    original = smoke.MAX_AUDIO_BYTES
    smoke.MAX_AUDIO_BYTES = 50
    try:
        with pytest.raises(smoke.SmokeBlocked, match="25 MiB"):
            smoke.inspect_audio(path)
    finally:
        smoke.MAX_AUDIO_BYTES = original


def test_valid_mp3_metadata_checked_without_decoding(monkeypatch, tmp_path):
    path = tmp_path / "private-file.mp3"
    path.write_bytes(b"fake")
    monkeypatch.setattr(smoke.shutil, "which", lambda n: "/fake/ffprobe")
    monkeypatch.setattr(smoke.subprocess, "run", lambda args, **kwargs: SimpleNamespace(
        returncode=0, stdout='{"format":{"duration":"14.500000"}}'
    ))
    data = smoke.inspect_audio(path)
    assert data == {"format": ".mp3", "size_bytes": 4, "duration_seconds": 14.5}
    assert "private-file" not in json.dumps(data)


@pytest.mark.parametrize("duration", ["0", "91", "not-a-number"])
def test_invalid_duration_blocked(monkeypatch, tmp_path, duration):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"bytes")
    monkeypatch.setattr(smoke.shutil, "which", lambda n: "/fake/ffprobe")
    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=json.dumps({"format": {"duration": duration}})
    ))
    with pytest.raises(smoke.SmokeBlocked):
        smoke.inspect_audio(path)


def test_ffprobe_failure_is_blocker(monkeypatch, tmp_path):
    path = tmp_path / "sample.wav"
    path.write_bytes(b"bytes")
    monkeypatch.setattr(smoke.shutil, "which", lambda n: "/fake/ffprobe")
    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=3, stdout=""
    ))
    with pytest.raises(smoke.SmokeBlocked, match="ffprobe"):
        smoke.inspect_audio(path)


class Word:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker


class Segment:
    def __init__(self, text="", speaker=None, words=None):
        self.text = text
        self.speaker = speaker
        self.words = words


def test_segment_evidence_complete_with_speaker_tags():
    result = smoke.evaluate_segments(
        [Segment("Здравствуйте", "SPEAKER_00", [Word(0.3, 1.2, "SPEAKER_00")])],
        require_diarization=True,
    )
    assert result["asr_verified"] and result["alignment_verified"] and result["diarization_verified"]
    assert result["speaker_count"] == 1
    assert result["result_contains_spoken_text"] is False


@pytest.mark.parametrize("segments,required,error", [
    ([], True, "zero"),
    ([Segment("")], True, "No recognized text"),
    ([Segment("Привет", words=[])], False, "word timestamps"),
    ([Segment("Привет", words=[Word(None, 0.2)])], False, "word timestamps"),
    ([Segment("Привет", words=[Word(0, 1)])], True, "speaker labels"),
])
def test_no_false_success_for_incomplete_pipeline(segments, required, error):
    with pytest.raises(smoke.SmokeFailed, match=error):
        smoke.evaluate_segments(segments, require_diarization=required)


def test_asr_alignment_without_diarization_is_evidence_but_not_full_pass():
    metrics = smoke.evaluate_segments([Segment("Привет", words=[Word(0, 1)])],
                                      require_diarization=False)
    assert metrics["asr_verified"]
    assert metrics["alignment_verified"]
    assert not metrics["diarization_verified"]


def test_environment_summary_excludes_secrets_and_paths():
    source = sample_report()
    source["hugging_face"]["token"] = "hf_SECRET_123"
    source["disk"]["path"] = "C:/Users/private"
    clean = smoke.sanitized_environment(source)
    assert clean["hf_token_present"] is True
    assert "hf_SECRET_123" not in json.dumps(clean)
    assert "Users" not in json.dumps(clean)


def test_atomic_report_has_no_input_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    data = {"status": "blocked", "note": "No model executed"}
    dest = smoke.save_report(data, root=tmp_path)
    assert dest.parent == tmp_path / "logs"
    assert json.loads(dest.read_text()) == data
    assert not list((tmp_path / "logs").glob("*.tmp"))


def test_diagnose_only_never_calls_heavy_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setattr(smoke, "build_report", lambda **k: sample_report())
    monkeypatch.setattr(smoke, "execute_pipeline", lambda *a, **k: pytest.fail("Must not infer"))
    result = smoke.main(["--diagnose-only"])
    assert result == smoke.EXIT_PASSED
    path = next((tmp_path / "logs").glob("smoke-*.json"))
    saved = json.loads(path.read_text())
    assert saved["status"] == "ready_for_manual_audio"
    assert saved["actual_inference_performed"] is False


def test_full_mode_blocked_by_hf_never_executes_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setattr(smoke, "build_report", lambda **k: sample_report(hf=False))
    monkeypatch.setattr(smoke, "execute_pipeline", lambda *a, **k: pytest.fail("Must not infer"))
    result = smoke.main(["--audio", str(tmp_path / "private.wav")])
    assert result == smoke.EXIT_BLOCKED
    path = next((tmp_path / "logs").glob("smoke-*.json"))
    saved = json.loads(path.read_text())
    assert saved["status"] == "blocked"
    assert saved["actual_inference_performed"] is False


def test_partial_mode_never_claims_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setattr(smoke, "build_report", lambda **k: sample_report(hf=False))
    monkeypatch.setattr(smoke, "inspect_audio", lambda p: {"format": ".wav", "duration_seconds": 10})
    monkeypatch.setattr(smoke, "execute_pipeline", lambda *a, **k: {
        "asr_verified": True, "alignment_verified": True, "diarization_verified": False
    })
    result = smoke.main(["--audio", str(tmp_path / "private.wav"), "--without-diarization"])
    assert result == smoke.EXIT_PARTIAL
    saved = json.loads(next((tmp_path / "logs").glob("smoke-*.json")).read_text())
    assert saved["status"] == "partial"
    assert saved["stages"]["diarization"] == "skipped"


def test_runtime_exception_redacted_in_report(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setattr(smoke, "build_report", lambda **k: sample_report())
    monkeypatch.setattr(smoke, "inspect_audio", lambda p: {"format": ".wav"})
    monkeypatch.setattr(smoke, "execute_pipeline",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("hf_super_private_secret")))
    result = smoke.main(["--audio", str(tmp_path / "private.wav")])
    assert result == smoke.EXIT_FAILED
    saved = json.loads(next((tmp_path / "logs").glob("smoke-*.json")).read_text())
    assert saved["status"] == "failed"
    assert "hf_super_private_secret" not in json.dumps(saved)
    assert saved["error_class"] == "RuntimeError"


def test_actual_pass_requires_full_pipeline_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    monkeypatch.setattr(smoke, "build_report", lambda **k: sample_report())
    monkeypatch.setattr(smoke, "inspect_audio", lambda p: {"format": ".mp3", "duration_seconds": 8})
    monkeypatch.setattr(smoke, "execute_pipeline", lambda *a, **k: {
        "asr_verified": True, "alignment_verified": True, "diarization_verified": True
    })
    assert smoke.main(["--audio", str(tmp_path / "short.mp3")]) == smoke.EXIT_PASSED
    saved = json.loads(next((tmp_path / "logs").glob("smoke-*.json")).read_text())
    assert saved["status"] == "passed"
    assert saved["actual_inference_performed"] is True


def test_execute_pipeline_reuses_single_existing_engine_with_exact_preset(monkeypatch, tmp_path):
    """Mock heavy imports but exercise the real P06 execute_pipeline wiring."""
    import sys
    from types import ModuleType

    made = []

    class Param:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Bundle(Param):
        def to_list(self):
            return [self.whisper, self.vad, self.diarization, self.bgm_separation]

    class FakeWhisperXInference:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.device = "cuda"
            self.whisperx_wrapper = None
            self.offloaded = False
            made.append(self)

        def run(self, audio, progress, output, add_timestamp, callback, *values, persist_settings=True, strict_vad=False):
            assert persist_settings is False
            assert strict_vad is True
            assert output == "txt"
            assert add_timestamp is False
            assert callback is None
            assert len(values) == 4
            whisper, vad, diarization, bgm = values
            assert whisper.model_size == "small"
            assert whisper.lang == "ru"  # canonical MVP code; the pipeline now accepts it
            assert whisper.is_translate is False
            assert whisper.compute_type == "int8"
            assert whisper.batch_size == 1
            assert whisper.enable_whisperx_alignment is True
            assert vad.vad_filter is True
            assert diarization.is_diarize is False
            assert diarization.use_whisperx_diarization is False
            assert bgm.is_separate_bgm is False
            assert self.device == "cpu"
            return [Segment("Привет", words=[Word(0.0, 1.0)])], 2.5

        def offload(self):
            self.offloaded = True

    gradio = ModuleType("gradio")
    gradio.Progress = lambda: object()
    whisper = ModuleType("whisper")
    whisper.tokenizer = SimpleNamespace(LANGUAGES={"ru": "russian"})
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_inference = ModuleType("modules.whisper.whisperx_inference")
    fake_inference.WhisperXInference = FakeWhisperXInference
    fake_data = ModuleType("modules.whisper.data_classes")
    for name in ("WhisperParams", "VadParams", "DiarizationParams", "BGMSeparationParams"):
        setattr(fake_data, name, Param)
    fake_data.TranscriptionPipelineParams = Bundle
    for name, module in (
        ("gradio", gradio), ("whisper", whisper), ("torch", torch),
        ("modules.whisper.whisperx_inference", fake_inference),
        ("modules.whisper.data_classes", fake_data),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    args = smoke.parse_args(["--audio", str(tmp_path / "fake.wav"), "--without-diarization"])
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    evidence = smoke.execute_pipeline(tmp_path / "fake.wav", args)
    assert len(made) == 1
    assert made[0].offloaded
    assert evidence["configuration"]["engine"] == "existing WhisperXInference.run"
    assert evidence["configuration"]["language"] == "ru"
    assert evidence["aligned_words"] == 1


def test_execute_pipeline_requires_cuda_when_explicitly_requested(monkeypatch, tmp_path):
    import sys
    from types import ModuleType
    gradio = ModuleType("gradio")
    whisper = ModuleType("whisper")
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    for name, module in (("gradio", gradio), ("whisper", whisper), ("torch", torch)):
        monkeypatch.setitem(sys.modules, name, module)
    # The engine imports also need stubs to reach the device guard.
    backend = ModuleType("modules.whisper.whisperx_inference")
    backend.WhisperXInference = object
    data = ModuleType("modules.whisper.data_classes")
    for cls in ("WhisperParams", "VadParams", "DiarizationParams",
                "BGMSeparationParams", "TranscriptionPipelineParams"):
        setattr(data, cls, object)
    monkeypatch.setitem(sys.modules, "modules.whisper.whisperx_inference", backend)
    monkeypatch.setitem(sys.modules, "modules.whisper.data_classes", data)
    args = smoke.parse_args(["--audio", str(tmp_path / "x.wav"), "--device", "cuda"])
    with pytest.raises(smoke.SmokeBlocked, match="CUDA"):
        smoke.execute_pipeline(tmp_path / "x.wav", args)



def test_two_voice_smoke_requires_two_distinct_speaker_ids():
    a = Segment("Добрый день", "SPEAKER_00", [Word(0.1, 0.4, "SPEAKER_00")])
    b = Segment("Здравствуйте", "SPEAKER_01", [Word(2.1, 2.7, "SPEAKER_01")])
    evidence = smoke.evaluate_segments([a, b], require_diarization=True, min_speakers=2)
    assert evidence["speaker_count"] == 2
    assert evidence["diarization_verified"] is True


def test_two_voice_smoke_rejects_one_speaker_without_false_success():
    a = Segment("Добрый день", "SPEAKER_00", [Word(0.1, 0.4, "SPEAKER_00")])
    b = Segment("Здравствуйте", "SPEAKER_00", [Word(2.1, 2.7, "SPEAKER_00")])
    with pytest.raises(smoke.SmokeFailed, match="at least 2 required"):
        smoke.evaluate_segments([a, b], require_diarization=True, min_speakers=2)


def test_smoke_min_speakers_requires_diarization_and_valid_range():
    assert smoke.parse_args(["--diagnose-only", "--min-speakers", "2"]).min_speakers == 2
    with pytest.raises(SystemExit):
        smoke.parse_args(["--diagnose-only", "--min-speakers", "0"])
    with pytest.raises(SystemExit):
        smoke.parse_args(["--diagnose-only", "--without-diarization", "--min-speakers", "2"])

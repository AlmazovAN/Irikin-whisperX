"""P07: offline, no audio/models/GPU/Gradio imported."""

from dataclasses import FrozenInstanceError
import sys
from types import ModuleType

import pytest

from modules.mvp import MVP_PRESET, build_mvp_pipeline_params

pytestmark = pytest.mark.fast


class FakeParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def fake_pipeline_models(monkeypatch):
    fake_data = ModuleType("modules.whisper.data_classes")
    for name in (
        "WhisperParams", "VadParams", "DiarizationParams",
        "BGMSeparationParams", "TranscriptionPipelineParams",
    ):
        setattr(fake_data, name, FakeParams)
    monkeypatch.setitem(sys.modules, "modules.whisper.data_classes", fake_data)
    return fake_data


def test_fixed_profile_is_immutable():
    assert (MVP_PRESET.model_size, MVP_PRESET.language, MVP_PRESET.task) == (
        "small", "ru", "transcribe",
    )
    assert (MVP_PRESET.vad_enabled, MVP_PRESET.alignment_enabled,
            MVP_PRESET.diarization_enabled) == (True, True, True)
    assert MVP_PRESET.output_format == "txt"
    with pytest.raises(FrozenInstanceError):
        MVP_PRESET.model_size = "large-v2"


def test_fresh_defaults_are_identical_for_cli_ui_and_future_jobs(fake_pipeline_models):
    # All consumers use the same entrypoint; no duplicate YAML or UI constants.
    ui = build_mvp_pipeline_params()
    job = build_mvp_pipeline_params()
    cli = build_mvp_pipeline_params()
    assert ui is not job and job is not cli
    for params in (ui, job, cli):
        assert params.whisper.model_size == "small"
        assert params.whisper.lang == "ru"
        assert params.whisper.is_translate is False
        assert params.whisper.compute_type == "int8"
        assert params.whisper.batch_size == 1
        assert params.whisper.enable_whisperx_alignment is True
        assert params.whisper.word_timestamps is False
        assert params.vad.vad_filter is True
        assert params.diarization.is_diarize is True
        assert params.diarization.use_whisperx_diarization is True
        assert params.diarization.diarization_device == "cpu"
        assert params.diarization.assign_word_speakers is True
        assert params.diarization.hf_token == ""
        assert params.bgm_separation.is_separate_bgm is False
    ui.whisper.model_size = "mutated"
    assert job.whisper.model_size == cli.whisper.model_size == "small"


def test_diagnostic_mode_is_explicit_not_mvp_acceptance(fake_pipeline_models):
    params = build_mvp_pipeline_params(diarization_enabled=False)
    assert params.whisper.enable_whisperx_alignment is True
    assert params.diarization.is_diarize is False
    assert params.diarization.use_whisperx_diarization is False
    assert params.diarization.assign_word_speakers is False


def test_cuda_is_explicit_not_guessed_from_amd(fake_pipeline_models):
    params = build_mvp_pipeline_params(device="cuda", compute_type="float16", batch_size=4)
    assert params.whisper.compute_type == "float16"
    assert params.diarization.diarization_device == "cuda"
    assert params.whisper.batch_size == 4


@pytest.mark.parametrize("options", [
    {"device": "mps"}, {"device": "auto"}, {"compute_type": "invalid"},
    {"compute_type": "float16", "device": "cpu"},
    {"batch_size": 0}, {"batch_size": 9}, {"batch_size": True},
    {"batch_size": 1.5}, {"diarization_enabled": "false"},
])
def test_bad_runtime_options_fail_before_ml_import(options):
    with pytest.raises(ValueError):
        build_mvp_pipeline_params(**options)


def test_pipeline_has_explicit_nonpersisting_mode():
    """Contract guard without importing the heavy legacy pipeline."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "modules/whisper/base_transcription_pipeline.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "BaseTranscriptionPipeline")
    run = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef)
               and node.name == "run")
    assert "persist_settings" in [a.arg for a in run.args.kwonlyargs]
    cache_conditional = [node for node in ast.walk(run)
                         if isinstance(node, ast.If)
                         and isinstance(node.test, ast.Name)
                         and node.test.id == "persist_settings"]
    assert len(cache_conditional) == 1
    assert any(isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute)
               and node.func.attr == "cache_parameters"
               for node in ast.walk(cache_conditional[0]))

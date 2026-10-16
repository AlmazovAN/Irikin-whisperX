"""P09: headless TranscriptionService, deterministic offline unit tests."""

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from types import ModuleType
import subprocess
import sys

import pytest

from modules.jobs.inputs import InputKind, ValidatedInput
from modules.jobs.transcription_service import (
    HeadlessProgress, TranscriptionService, TranscriptionServiceError,
)

pytestmark = pytest.mark.fast


@pytest.fixture(autouse=True)
def stub_heavy_model_schemas(monkeypatch):
    """Exercise real P07 preset builder without installing ML packages in CI."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_fixture_not_a_real_secret")
    fake_module = ModuleType("modules.whisper.data_classes")

    class FakeParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Bundle(FakeParams):
        def to_list(self):
            return [self.whisper, self.vad, self.diarization, self.bgm_separation]

    for name in ("WhisperParams", "VadParams", "DiarizationParams", "BGMSeparationParams"):
        setattr(fake_module, name, FakeParams)
    fake_module.TranscriptionPipelineParams = Bundle
    monkeypatch.setitem(sys.modules, "modules.whisper.data_classes", fake_module)


@dataclass
class FakeWord:
    start: float = 0.1
    end: float = 0.4
    word: str = "Привет"
    speaker: str | None = "SPEAKER_00"


@dataclass
class FakeSegment:
    start: float = 0.1
    end: float = 0.4
    text: str = "Привет"
    words: list | None = None
    speaker: str | None = "SPEAKER_00"


def spoken(*, speaker: bool = True, aligned: bool = True):
    return [FakeSegment(
        words=[FakeWord(speaker="SPEAKER_00" if speaker else None)] if aligned else [],
        speaker="SPEAKER_00" if speaker else None,
    )]


@pytest.fixture
def valid_source(tmp_path):
    path = tmp_path / "private.wav"
    path.write_bytes(b"private fake test audio")
    return ValidatedInput(
        source_kind=InputKind.LOCAL_FILE, path=path.resolve(), safe_filename="private.wav",
        storage_filename="uuid-private.wav", size_bytes=path.stat().st_size,
        duration_seconds=0.4, codec_name="pcm_s16le",
    )


class FakePipeline:
    def __init__(self, segments=None, elapsed=1.25, fail=None, block=None):
        self.segments = segments if segments is not None else spoken()
        self.elapsed = elapsed
        self.fail = fail
        self.block = block
        self.device = "unassigned"
        self.calls = []
        self.offloads = 0
        self.whisperx_wrapper = None

    def run(self, path, progress, file_format, add_timestamp,
            callback, *config, persist_settings=True, strict_vad=False):
        self.calls.append((path, progress, file_format, add_timestamp, callback, config, persist_settings))
        assert strict_vad is True
        progress(0, desc="Filtering silent parts from audio")
        progress(0.95, desc="Aligning")
        if self.block is not None:
            ready, proceed = self.block
            ready.set()
            proceed.wait(timeout=2)
        if self.fail:
            raise self.fail
        if callback:
            callback(0.5)
        return self.segments, self.elapsed

    def offload(self):
        self.offloads += 1


def test_headless_service_calls_only_existing_pipeline(valid_source):
    pipeline = FakePipeline()
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return pipeline

    events = []
    service = TranscriptionService(pipeline_factory=factory)
    result = service.transcribe(valid_source, progress_callback=events.append)
    assert len(factory_calls) == 1
    assert pipeline.device == "cpu"
    assert result.segments[0] is pipeline.segments[0]
    assert result.elapsed_seconds == 1.25
    assert (result.model_size, result.language) == ("small", "ru")
    assert (result.input_name, result.diarization_requested, result.diagnostic_only) == (
        "private.wav", True, False)
    assert events == [0.5]
    path, progress, ext, stamp, cb, params, persist = pipeline.calls[0]
    assert path == str(valid_source.path)
    assert isinstance(progress, HeadlessProgress)
    assert (ext, stamp, persist) == ("txt", False, False)
    assert len(params) == 4
    whisper, vad, diar, bgm = params
    assert whisper.model_size == "small" and whisper.lang == "ru"
    assert whisper.enable_whisperx_alignment
    assert vad.vad_filter and diar.is_diarize and not bgm.is_separate_bgm
    assert pipeline.offloads == 1


def test_same_service_reuses_one_inference_owner(valid_source):
    pipeline = FakePipeline()
    used = []
    service = TranscriptionService(pipeline_factory=lambda: used.append(True) or pipeline)
    service.transcribe(valid_source)
    service.transcribe(valid_source)
    assert len(used) == 1
    assert pipeline.offloads == 2
    assert len(pipeline.calls) == 2


def test_explicit_diagnostic_mode_is_not_accepted_mvp(valid_source):
    fake = FakePipeline(segments=spoken(speaker=False))
    result = TranscriptionService(pipeline_factory=lambda: fake).transcribe(
        valid_source, diarization_enabled=False,
    )
    assert result.diagnostic_only
    assert result.diarization_requested is False
    assert fake.calls[0][5][2].is_diarize is False


@pytest.mark.parametrize("value,code", [
    ([], "EMPTY_TRANSCRIPT"),
    ([FakeSegment(text="", words=[])], "EMPTY_TRANSCRIPT"),
    ([FakeSegment(start=None, words=[FakeWord()])], "INVALID_SEGMENTS"),
    ([FakeSegment(end=float("nan"), words=[FakeWord()])], "INVALID_SEGMENTS"),
    (spoken(aligned=False), "ALIGNMENT_MISSING"),
    (spoken(speaker=False), "DIARIZATION_MISSING"),
])
def test_incomplete_output_is_not_claimed_success(valid_source, value, code):
    pipeline = FakePipeline(segments=value)
    with pytest.raises(TranscriptionServiceError) as exc:
        TranscriptionService(pipeline_factory=lambda: pipeline).transcribe(valid_source)
    assert exc.value.code == code
    assert pipeline.offloads == 1


def test_pipeline_failure_is_redacted(valid_source):
    private = "hf_secret_do_not_print_or_log"
    pipeline = FakePipeline(fail=RuntimeError(private))
    with pytest.raises(TranscriptionServiceError) as exc:
        TranscriptionService(pipeline_factory=lambda: pipeline).transcribe(valid_source)
    assert exc.value.code == "PIPELINE_FAILED"
    assert private not in str(exc.value)
    assert pipeline.offloads == 1


def test_changed_or_missing_input_rejected_before_inference(valid_source):
    pipeline = FakePipeline()
    service = TranscriptionService(pipeline_factory=lambda: pipeline)
    valid_source.path.write_bytes(b"changed")
    with pytest.raises(TranscriptionServiceError) as exc:
        service.transcribe(valid_source)
    assert exc.value.code == "INPUT_CHANGED"
    assert pipeline.calls == []


def test_runtime_options_checked_before_model_creation(valid_source):
    pipeline = FakePipeline()
    service = TranscriptionService(pipeline_factory=lambda: pipeline)
    with pytest.raises(ValueError):
        service.transcribe(valid_source, device="cpu", compute_type="float16")
    assert pipeline.calls == []


def test_busy_rejected_without_second_inference(valid_source):
    ready, proceed = Event(), Event()
    pipeline = FakePipeline(block=(ready, proceed))
    service = TranscriptionService(pipeline_factory=lambda: pipeline)
    completed = []
    thread = Thread(target=lambda: completed.append(service.transcribe(valid_source)), daemon=True)
    thread.start()
    try:
        assert ready.wait(timeout=2)
        with pytest.raises(TranscriptionServiceError) as exc:
            service.transcribe(valid_source)
        assert exc.value.code == "BUSY"
        assert len(pipeline.calls) == 1
    finally:
        proceed.set()
        thread.join(timeout=2)
    assert len(completed) == 1


def test_invalid_elapsed_is_an_error(valid_source):
    fake = FakePipeline(elapsed=float("nan"))
    with pytest.raises(TranscriptionServiceError) as exc:
        TranscriptionService(pipeline_factory=lambda: fake).transcribe(valid_source)
    assert exc.value.code == "INVALID_DURATION"


def test_import_has_no_heavy_import_side_effects():
    script = (
        "import modules.jobs.transcription_service; import sys; "
        "assert 'gradio' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert 'whisperx' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr



def test_missing_hf_token_blocks_before_heavy_model(valid_source, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    made = []
    service = TranscriptionService(pipeline_factory=lambda: made.append(True))
    with pytest.raises(TranscriptionServiceError) as exc:
        service.transcribe(valid_source)
    assert exc.value.code == "HF_TOKEN_MISSING"
    assert "HF_TOKEN" in str(exc.value)
    assert not made
    assert service._pipeline is None


def test_diagnostic_asr_only_does_not_need_hf_token(valid_source, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fake = FakePipeline(segments=spoken(speaker=False))
    result = TranscriptionService(pipeline_factory=lambda: fake).transcribe(
        valid_source, diarization_enabled=False,
    )
    assert result.diagnostic_only


@pytest.mark.parametrize("stage_code", [
    "HF_MODEL_ACCESS", "ALIGNMENT_MODEL_UNAVAILABLE", "ALIGNMENT_FAILED",
    "CUDA_UNAVAILABLE", "DIARIZATION_FAILED",
])
def test_stage_specific_failures_keep_actionable_code_without_secrets(
    valid_source, stage_code,
):
    from modules.diarize.diagnostics import ModelStageError
    original_secret = "hf_super_private_secret_do_not_echo"
    fake = FakePipeline(fail=ModelStageError(stage_code, "Stage failed; check model access"))
    with pytest.raises(TranscriptionServiceError) as exc:
        TranscriptionService(pipeline_factory=lambda: fake).transcribe(valid_source)
    assert exc.value.code == stage_code
    assert original_secret not in str(exc.value)
    assert fake.offloads == 1


def test_two_speakers_fixture_keeps_independent_labels(valid_source):
    utterances = [
        FakeSegment(start=0, end=0.5, text="Добрый день",
                    speaker="SPEAKER_00", words=[FakeWord(start=0.1, end=0.4, speaker="SPEAKER_00")]),
        FakeSegment(start=2, end=2.5, text="Здравствуйте",
                    speaker="SPEAKER_01", words=[FakeWord(start=2.1, end=2.4, speaker="SPEAKER_01")]),
    ]
    pipeline = FakePipeline(segments=utterances)
    result = TranscriptionService(pipeline_factory=lambda: pipeline).transcribe(valid_source)
    assert {seg.speaker for seg in result.segments} == {"SPEAKER_00", "SPEAKER_01"}
    assert result.segments[0].words[0].speaker == "SPEAKER_00"
    assert result.segments[1].words[0].speaker == "SPEAKER_01"


def test_real_transcription_adapter_emits_stages_without_changing_asr_callback(valid_source):
    """Check the actual TranscriptionService boundary, with a fake pipeline."""
    class EmittingPipeline(FakePipeline):
        def run(self, path, progress, output_format, timestamp, callback, *params,
                persist_settings=True, strict_vad=False):
            assert strict_vad
            progress(0.0, desc="Filtering silent parts from audio..")
            progress(0.0, desc="Transcribing with WhisperX..")
            if callback:
                callback(0.3)
            progress(0.95, desc="Aligning with WhisperX..")
            progress(0.99, desc="Diarizing speakers..")
            progress(1.0, desc="Finished.")
            return self.segments, self.elapsed
    pipeline = EmittingPipeline()
    stages, asr_updates = [], []
    service = TranscriptionService(pipeline_factory=lambda: pipeline)
    result = service.transcribe(
        valid_source, progress_callback=asr_updates.append,
        stage_callback=stages.append,
    )
    assert stages == ["VAD", "ASR", "ALIGN", "DIARIZE"]
    assert asr_updates == [0.3]
    assert result.model_size == "small"
    assert pipeline.offloads == 1


def test_noncallable_stage_callback_is_rejected(valid_source):
    service = TranscriptionService(pipeline_factory=FakePipeline)
    with pytest.raises(TypeError, match="stage_callback"):
        service.transcribe(valid_source, stage_callback=123)

"""P10: deterministic VAD timeline and Russian-only MVP pipeline regressions.

Pure stdlib tests: neither WhisperX nor faster-whisper/torch/gradio are imported.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from modules.vad.timestamps import SpeechTimeMap, restore_original_timestamps

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[2]
RATE = 16000


@dataclass
class Word:
    start: float | None
    end: float | None
    word: str = "привет"


@dataclass
class Segment:
    start: float | None = None
    end: float | None = None
    text: str = ""
    words: list[Word] | None = None
    speaker: str | None = None


def two_chunks():
    # VAD extracted speech at 0-2s and 5-8s; compressed audio is 0-5s.
    return [{"start": 0, "end": 2 * RATE},
            {"start": 5 * RATE, "end": 8 * RATE}]


def test_no_shift_in_first_chunk_and_full_gap_in_second():
    mapper = SpeechTimeMap(two_chunks())
    assert mapper.original_time(0) == 0
    assert mapper.original_time(1.5) == 1.5
    assert mapper.original_time(2.5) == 5.5
    assert mapper.original_time(4.5, endpoint="end") == 7.5
    assert mapper.original_time(5, endpoint="end") == 8


def test_seam_start_and_end_map_to_opposite_sides_of_pause():
    mapper = SpeechTimeMap(two_chunks())
    assert mapper.original_time(2, endpoint="end") == 2
    assert mapper.original_time(2, endpoint="start") == 5


def test_restored_segment_with_no_words_spans_actual_pause():
    result = restore_original_timestamps([Segment(1.9, 2.2, "До и после")], two_chunks())
    assert len(result) == 1
    assert result[0].start == pytest.approx(1.9)
    assert result[0].end == pytest.approx(5.2)


def test_word_endpoints_across_cut_are_not_collapsed_into_one_chunk():
    seg = Segment(1.8, 2.5, "два слова", words=[
        Word(1.85, 2, "до"), Word(2, 2.4, "после"),
    ])
    mapped = restore_original_timestamps([seg], two_chunks())[0]
    assert mapped.words[0].start == pytest.approx(1.85)
    assert mapped.words[0].end == pytest.approx(2.0)
    assert mapped.words[1].start == pytest.approx(5.0)
    assert mapped.words[1].end == pytest.approx(5.4)
    assert mapped.start == pytest.approx(1.85)
    assert mapped.end == pytest.approx(5.4)


def test_missing_word_timestamps_do_not_crash_or_destroy_text():
    seg = Segment(2.1, 3.0, "с неполным alignment",
                  words=[Word(None, None), Word(2.2, 2.5)])
    restore_original_timestamps([seg], two_chunks())
    assert seg.text == "с неполным alignment"
    assert seg.words[0].start is None
    assert seg.words[1].start == pytest.approx(5.2)
    assert seg.start == pytest.approx(5.2)


def test_hour_long_recordings_preserve_sample_boundaries():
    samples = [{"start": 3600 * RATE, "end": 3603 * RATE},
               {"start": 7200 * RATE, "end": 7202 * RATE}]
    mapper = SpeechTimeMap(samples)
    assert mapper.original_time(1.25) == pytest.approx(3601.25)
    assert mapper.original_time(4.125) == pytest.approx(7201.125)


def test_contiguous_chunks_have_no_artificial_gap():
    mapper = SpeechTimeMap([{"start": 0, "end": RATE}, {"start": RATE, "end": 2 * RATE}])
    assert mapper.original_time(1.0) == pytest.approx(1.0)
    assert mapper.original_time(1.0, endpoint="end") == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [
    [], [{"start": 0, "end": 0}], [{"start": -1, "end": 5}],
    [{"start": 10, "end": 9}],
    [{"start": 0, "end": 100}, {"start": 50, "end": 200}],
    [{"start": 0.5, "end": 100}], [{"start": True, "end": 100}],
])
def test_invalid_vad_chunk_map_rejected(bad):
    with pytest.raises(ValueError):
        SpeechTimeMap(bad)


@pytest.mark.parametrize("stamp", [-0.01, float("nan"), float("inf"), 5.25, None])
def test_invalid_timestamps_never_silently_shift(stamp):
    with pytest.raises(ValueError):
        SpeechTimeMap(two_chunks()).original_time(stamp)


def test_empty_segments_do_not_require_vad_map():
    assert restore_original_timestamps([], [], RATE) == []


def test_mapper_rejects_wrong_rate():
    with pytest.raises(ValueError):
        SpeechTimeMap(two_chunks(), sample_rate=0)


def _extract_pipeline_run():
    """Compile the *real* pipeline.run method with lightweight globals.

    This tests order and data flow instead of duplicating run's implementation.
    The file is parsed, not imported, so CI needs no model stack.
    """
    source = ROOT / "modules/whisper/base_transcription_pipeline.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    parent = next(node for node in module.body if isinstance(node, ast.ClassDef)
                  and node.name == "BaseTranscriptionPipeline")
    func = next(node for node in parent.body if isinstance(node, ast.FunctionDef)
                and node.name == "run")
    ast_module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
              func],
        type_ignores=[],
    )
    ast.fix_missing_locations(ast_module)
    globals_ = {
        "gr": SimpleNamespace(Progress=lambda: lambda *a, **k: None),
        "time": time,
        "deepcopy": lambda obj: obj,
        "VadOptions": lambda **kw: SimpleNamespace(**kw),
        "TranscriptionPipelineParams": SimpleNamespace(from_list=lambda a: a[0]),
        "Segment": Segment,
        "validate_audio": lambda audio: True,
        "logger": SimpleNamespace(info=lambda *a: None),
    }
    exec(compile(ast_module, str(source), "exec"), globals_)
    return globals_["run"]


class Audio:
    def __init__(self, label, size):
        self.label = label
        self.size = size


def _run_fixture(*, speech=True, strict=True, alignment_language="en"):
    events = []
    orig, trimmed = Audio("original", 90000), Audio("vad_filtered", 40000 if speech else 0)
    params = SimpleNamespace(
        whisper=SimpleNamespace(lang="ru", is_translate=False,
                                enable_whisperx_alignment=True, enable_offload=False),
        vad=SimpleNamespace(vad_filter=True, threshold=0.5,
                            min_speech_duration_ms=250, max_speech_duration_s=100,
                            min_silence_duration_ms=2000, speech_pad_ms=400),
        diarization=SimpleNamespace(is_diarize=True, use_whisperx_diarization=True,
                                     enable_offload=False),
        bgm_separation=SimpleNamespace(is_separate_bgm=False),
    )

    class VAD:
        def run(self, *, audio, vad_parameters, progress):
            assert audio is orig
            events.append("vad")
            return trimmed, two_chunks() if speech else []

        def restore_speech_timestamps(self, *, segments, speech_chunks):
            events.append("restore")
            return restore_original_timestamps(segments, speech_chunks)

    class Wrapper:
        def transcribe(self, audio, whisper_params, device, progress, progress_callback):
            events.append("asr")
            assert audio is (trimmed if speech else orig)
            assert whisper_params.lang == "ru"
            assert whisper_params.is_translate is False
            return [Segment(2.25, 2.75, "Привет")], alignment_language, 1.0

        def align(self, *, segments, language, audio, device):
            events.append("align")
            assert language == "ru"  # ignore model's reported 'en'
            assert audio is orig
            assert segments[0].start == pytest.approx(5.25 if speech else 2.25)
            assert segments[0].end == pytest.approx(5.75 if speech else 2.75)
            segments[0].words = [Word(5.3, 5.6)]
            return segments

        def diarize(self, *, audio, segments, diarization_params):
            events.append("diarization")
            assert audio is orig
            segments[0].speaker = "SPEAKER_00"
            return segments

    wrapper = Wrapper()
    pipeline = SimpleNamespace(
        _check_cancelled=lambda: None,
        vad=VAD(),
        whisperx_wrapper=wrapper,
        validate_gradio_values=lambda p: p,
        device="cpu",
        offload=lambda: None,
        cache_parameters=lambda **kw: events.append("cache"),
    )
    return _extract_pipeline_run()(pipeline, orig, lambda *a, **k: None, "txt",
                                   False, None, params, persist_settings=False,
                                   strict_vad=strict), events


def test_pipeline_restores_before_alignment_and_diarization_on_original_audio():
    (segments, duration), events = _run_fixture()
    assert segments[0].start == pytest.approx(5.25)
    assert segments[0].end == pytest.approx(5.75)
    assert segments[0].speaker == "SPEAKER_00"
    assert events == ["vad", "asr", "restore", "align", "diarization"]
    assert duration >= 0


def test_strict_mvp_vad_silence_skips_asr_alignment_and_diarization():
    (segments, duration), events = _run_fixture(speech=False, strict=True)
    assert len(segments) == 1 and not segments[0].text
    assert events == ["vad"]


def test_legacy_vad_silence_does_not_change_existing_fallback():
    # The existing legacy UI remains permissive; strictness is MVP opt-in.
    (segments, duration), events = _run_fixture(speech=False, strict=False)
    assert events[0:2] == ["vad", "asr"]
    assert "restore" not in events


def test_actual_whisperx_asr_method_passes_ru_and_transcribe():
    """Run the real WhisperXInference._transcribe_with_fallback with a fake model."""
    source = ROOT / "modules/whisper/whisperx_inference.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef)
               and n.name == "WhisperXInference")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_transcribe_with_fallback")
    test_ast = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        method,
    ], type_ignores=[])
    ast.fix_missing_locations(test_ast)
    scope = {}
    exec(compile(test_ast, str(source), "exec"), scope)
    calls = []

    def infer(audio, **kwargs):
        calls.append(kwargs)
        return {"segments": []}

    owner = SimpleNamespace(model=SimpleNamespace(transcribe=infer), device="cpu")
    params = SimpleNamespace(chunk_length=30, is_translate=False, batch_size=1, lang="ru")
    value = scope["_transcribe_with_fallback"](owner, "fake numpy audio", params)
    assert value == {"segments": []}
    assert calls == [{
        "batch_size": 1, "language": "ru", "task": "transcribe", "chunk_size": 30,
    }]

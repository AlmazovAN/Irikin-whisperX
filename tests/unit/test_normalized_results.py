"""PR13: standalone JSON/TXT schema, speaker, confidence and legacy regressions.

No model downloads, Gradio session or heavy ML imports are required.
"""

from dataclasses import dataclass, field
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from modules.jobs.results import (
    ResultFormatError, SCHEMA_VERSION, build_result_payload,
    normalize_segments, render_json, render_txt, write_result_files,
)

pytestmark = pytest.mark.fast
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Word:
    word: str
    start: float | None = None
    end: float | None = None
    probability: float | None = None
    speaker: str | None = None


@dataclass
class Segment:
    start: float | None
    end: float | None
    text: str | None
    speaker: str | None = None
    words: list | None = None


@dataclass
class Result:
    segments: tuple
    elapsed_seconds: float = 3.25
    input_name: str = "встреча.wav"
    model_size: str = "small"
    language: str = "ru"
    diarization_requested: bool = True


@pytest.fixture
def result():
    return Result(segments=(
        Segment(
            start=0.0, end=1.25, text="SPEAKER_00|Добрый день",
            speaker="SPEAKER_00",
            words=[
                Word(" Добрый", 0.1, 0.45, 0.0, "SPEAKER_00"),
                Word(" день", 0.46, 1.15, 0.95, "SPEAKER_00"),
            ],
        ),
        Segment(
            start=4.0, end=5.5, text="SPEAKER_01|Здравствуйте",
            speaker="SPEAKER_01",
            words=[Word("Здравствуйте", 4.1, 5.2, 0.71, "SPEAKER_01")],
        ),
    ))


def test_two_speakers_and_words_become_canonical_json(result):
    payload = build_result_payload(result)
    assert payload["schema_version"] == SCHEMA_VERSION == 1
    assert payload["language"] == "ru"
    assert payload["model_size"] == "small"
    assert payload["source_name"] == "встреча.wav"
    assert payload["diagnostic_only"] is False
    assert len(payload["segments"]) == 2
    a, b = payload["segments"]
    assert a["speaker"] == "SPEAKER_00"
    assert b["speaker"] == "SPEAKER_01"
    assert a["text"] == "Добрый день"
    assert b["text"] == "Здравствуйте"
    assert a["words"][0] == {
        "word": "Добрый", "start": 0.1, "end": 0.45,
        "speaker": "SPEAKER_00", "confidence": 0.0,
    }
    assert b["words"][0]["confidence"] == 0.71
    assert set(a) == {"start", "end", "text", "speaker", "words"}


def test_no_mutation_to_legacy_writer_segments(result):
    old = result.segments[0]
    original_text = old.text
    original_word = old.words[0].word
    first = normalize_segments(result.segments)
    first[0]["text"] = "tampered"
    first[0]["words"][0]["word"] = "tampered"
    assert old.text == original_text
    assert old.words[0].word == original_word
    again = normalize_segments(result.segments)
    assert again[0]["text"] == "Добрый день"


def test_existing_diarization_prefix_without_explicit_speaker_is_recovered():
    normalized = normalize_segments([
        Segment(1, 2, "SPEAKER_07|Текст", None),
    ])
    assert normalized == [{
        "start": 1.0, "end": 2.0, "text": "Текст", "speaker": "SPEAKER_07",
    }]


def test_mismatching_label_does_not_rewrite_speaker_or_text():
    original = Segment(1, 2, "SPEAKER_03|Не совпадает", "SPEAKER_04")
    result = normalize_segments([original])[0]
    assert result["speaker"] == "SPEAKER_04"
    assert result["text"] == "SPEAKER_03|Не совпадает"


def test_legacy_none_prefix_and_unassigned_speaker():
    assert normalize_segments([Segment(1, 2, "None|Без спикера")])[0] == {
        "start": 1.0, "end": 2.0, "text": "Без спикера", "speaker": None,
    }


def test_infer_single_word_speaker_but_not_multiple_labels():
    a = Segment(1, 2, "Привет", words=[Word("Привет", 1.1, 1.9, speaker="SPEAKER_00")])
    b = Segment(3, 4, "Ответ", words=[
        Word("Первый", 3.1, 3.3, speaker="SPEAKER_00"),
        Word("Второй", 3.4, 3.8, speaker="SPEAKER_01"),
    ])
    normalized = normalize_segments([a, b])
    assert normalized[0]["speaker"] == "SPEAKER_00"
    assert normalized[1]["speaker"] is None
    assert [w["speaker"] for w in normalized[1]["words"]] == [
        "SPEAKER_00", "SPEAKER_01",
    ]


def test_unaligned_words_preserved_without_fake_timestamps():
    result = normalize_segments([Segment(1, 2, "Привет", words=[
        Word(" Привет", None, None, None),
        {"word": "!", "start": 1.6, "probability": 0.0},
    ])])
    assert result[0]["words"] == [
        {"word": "Привет"},
        {"word": "!", "start": 1.6, "confidence": 0.0},
    ]


def test_native_whisperx_dict_shape_and_zero_probability():
    segments = [{"start": 0, "end": 1, "text": "Русский",
                 "speaker": "SPEAKER_00",
                 "words": [{"word": "Русский", "start": 0, "end": 1,
                            "score": 0.0, "probability": 0.78}]}]
    normalized = normalize_segments(segments)
    assert normalized[0]["words"][0]["confidence"] == 0.78
    native = [{"start": 0, "end": 1, "text": "Русский",
               "words": [{"word": "Русский", "start": 0, "end": 1,
                          "score": 0.0}]}]
    assert normalize_segments(native)[0]["words"][0]["confidence"] == 0.0


def test_json_is_valid_utf8_non_escaped_russian_and_roundtrips(result):
    raw = render_json(build_result_payload(result))
    assert "Здравствуйте" in raw
    assert "\\u0417" not in raw
    parsed = json.loads(raw)
    assert parsed["segments"][0]["speaker"] == "SPEAKER_00"
    assert "hf_token" not in raw.lower()


def test_txt_is_readable_with_original_timestamps_and_speakers(result):
    txt = render_txt(build_result_payload(result))
    assert txt == (
        "[00:00:00.000 --> 00:00:01.250] SPEAKER_00: Добрый день\n"
        "[00:00:04.000 --> 00:00:05.500] SPEAKER_01: Здравствуйте\n"
    )
    assert "SPEAKER_00|" not in txt


def test_txt_without_speaker_and_long_timestamps():
    data = {"segments": [{"start": 3600.05, "end": 3603.3,
                          "text": "Длинная запись\n  с переносом", "speaker": None}]}
    assert render_txt(data) == (
        "[01:00:00.050 --> 01:00:03.300] Длинная запись с переносом\n"
    )


def test_empty_placeholders_excluded_but_not_false_success():
    assert normalize_segments([Segment(None, None, None), Segment(None, None, "")]) == []
    with pytest.raises(ResultFormatError) as exc:
        build_result_payload(Result(segments=(Segment(None, None, None),)))
    assert exc.value.code == "EMPTY_TRANSCRIPT"


@pytest.mark.parametrize("segment", [
    Segment(-1, 2, "Тест"),
    Segment(2, 1, "Тест"),
    Segment(float("nan"), 2, "Тест"),
    Segment(0, float("inf"), "Тест"),
    Segment("00:01", 2, "Тест"),
    Segment(True, 2, "Тест"),
])
def test_invalid_segment_times_are_rejected(segment):
    with pytest.raises(ResultFormatError) as exc:
        normalize_segments([segment])
    assert exc.value.code == "INVALID_TIMESTAMP"


@pytest.mark.parametrize("word", [
    Word("нет", start=2, end=1),
    Word("нет", start=float("nan"), end=1),
    Word("нет", start=0, end=1, probability=-0.2),
    Word("нет", start=0, end=1, probability=1.2),
])
def test_invalid_word_timestamps_or_confidence_rejected(word):
    with pytest.raises(ResultFormatError) as exc:
        normalize_segments([Segment(0, 3, "Слова", words=[word])])
    assert exc.value.code in ("INVALID_TIMESTAMP", "INVALID_CONFIDENCE")


def test_diagnostic_only_flag_is_explicit(result):
    result.diarization_requested = False
    payload = build_result_payload(result)
    assert payload["diarization_requested"] is False
    assert payload["diagnostic_only"] is True


def test_basename_only_no_private_paths_or_nonwhitelisted_metadata(result):
    result.input_name = r"C:\Users\Someone\Private\meeting.wav"
    result.hf_token = "hf_DO_NOT_EXPORT"
    payload = build_result_payload(result)
    raw = render_json(payload)
    assert payload["source_name"] == "meeting.wav"
    assert r"C:\Users" not in raw
    assert result.hf_token not in raw
    assert "hf_token" not in raw


def test_export_writes_two_files_to_existing_directory(result, tmp_path):
    files = write_result_files(result, tmp_path, stem="job_a123")
    assert files.json_path.name == "job_a123.json"
    assert files.txt_path.name == "job_a123.txt"
    assert json.loads(files.json_path.read_text(encoding="utf-8"))["language"] == "ru"
    assert "SPEAKER_01: Здравствуйте" in files.txt_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_result_files(result, tmp_path, stem="job_a123")
    assert len(list(tmp_path.iterdir())) == 2


def test_export_generates_unique_names(result, tmp_path):
    a, b = write_result_files(result, tmp_path), write_result_files(result, tmp_path)
    assert a.json_path != b.json_path
    assert a.txt_path != b.txt_path


def test_export_never_overwrites_txt_and_rolls_back_new_json(result, tmp_path):
    existing = tmp_path / "job_42.txt"
    existing.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_result_files(result, tmp_path, stem="job_42")
    assert existing.read_text(encoding="utf-8") == "existing"
    assert not (tmp_path / "job_42.json").exists()


@pytest.mark.parametrize("unsafe", ["../escape", "a/b", r"a\b", ".", "..", "", "a.name"])
def test_export_rejects_untrusted_filename(result, tmp_path, unsafe):
    with pytest.raises(ResultFormatError) as exc:
        write_result_files(result, tmp_path, stem=unsafe)
    assert exc.value.code == "INVALID_FILENAME"


def test_export_rejects_nonexisting_output_directory(result, tmp_path):
    with pytest.raises(ResultFormatError) as exc:
        write_result_files(result, tmp_path / "missing", stem="job1")
    assert exc.value.code == "INVALID_OUTPUT_DIR"


def test_legacy_segment_declares_speaker_only_once():
    tree = ast.parse((ROOT / "modules/whisper/data_classes.py").read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "Segment")
    speakers = [x for x in cls.body if isinstance(x, ast.AnnAssign)
                and isinstance(x.target, ast.Name) and x.target.id == "speaker"]
    assert len(speakers) == 1


def test_actual_legacy_whisperx_converter_preserves_zero_confidence():
    """Regression: score=0.0 must not be dropped in _segments_from_dict."""
    path = ROOT / "modules/whisper/whisperx_wrapper.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and x.name == "WhisperXWrapper")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                  and x.name == "_segments_from_dict")
    method.decorator_list = []
    scope = {
        "Word": lambda **kw: SimpleNamespace(**kw),
        "Segment": lambda **kw: SimpleNamespace(**kw),
    }
    compiled = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        method,
    ], type_ignores=[])
    ast.fix_missing_locations(compiled)
    exec(compile(compiled, str(path), "exec"), scope)
    raw = [{"text": "SPEAKER_00|Привет", "speaker": "SPEAKER_00",
            "start": 1, "end": 2, "words": [
                {"word": "Привет", "start": 1, "end": 2, "score": 0.0},
            ]}]
    segments = scope["_segments_from_dict"](raw, prefix_speaker=False)
    assert segments[0].words[0].probability == 0.0
    # Original legacy prefixed text is still available to Advanced UI.
    assert segments[0].text == "SPEAKER_00|Привет"
    assert normalize_segments(segments)[0]["text"] == "Привет"


def test_import_is_lightweight_without_gradio_or_whisperx():
    script = ("import modules.jobs.results; import sys; "
              "assert 'torch' not in sys.modules; "
              "assert 'gradio' not in sys.modules; "
              "assert 'whisperx' not in sys.modules")
    done = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=10)
    assert done.returncode == 0, done.stderr

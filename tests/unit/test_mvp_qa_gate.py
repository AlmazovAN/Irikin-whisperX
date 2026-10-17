"""P25: QA release logic with real SQLite/files and a fake inference engine.

These tests are *not* real Windows/Pyannote/browser acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import wave

import pytest

from scripts import qa_mvp, mvp_release_gate
from modules.jobs.worker import JobWorker as RealWorker

pytestmark = pytest.mark.fast


@dataclass
class Word:
    word: str = "Привет"
    start: float = 0.1
    end: float = 0.6
    speaker: str = "SPEAKER_00"


@dataclass
class Segment:
    text: str = "Привет"
    start: float = 0.0
    end: float = 1.0
    speaker: str = "SPEAKER_00"
    words: tuple = (Word(),)


@dataclass
class Result:
    segments: tuple = (Segment(),)
    input_name: str = "example.wav"
    elapsed_seconds: float = 0.2
    model_size: str = "small"
    language: str = "ru"
    diarization_requested: bool = True


class FakeASR:
    def transcribe(self, source):
        if "two" in source.safe_filename:
            other = Segment(
                text="Ответ", start=1.1, end=2.0, speaker="SPEAKER_01",
                words=(Word("Ответ", 1.2, 1.7, "SPEAKER_01"),),
            )
            return Result(segments=(Segment(), other), input_name=source.safe_filename)
        return Result(input_name=source.safe_filename)


def _wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 5000)
    return path


def _report():
    return {
        "system": {"name": "Windows"},
        "python": {"version": "3.11.5", "supported_by_readme": True},
        "ram": {"total_gib": 8},
        "ffmpeg": {"status": "ok"},
        "ffprobe": {"status": "ok"},
        "torch": {"status": "ok", "cuda_available": False},
        "hugging_face": {"token_present": True, "secret": "hf_HIDDEN"},
        "packages": {name: {"status": "installed", "version": "candidate"}
                     for name in qa_mvp.REQUIRED_PACKAGES},
    }


def test_preflight_warnings_do_not_claim_real_model_success():
    report = _report()
    assert qa_mvp.check_preflight(report) == []
    report["system"]["name"] = "Linux"
    report["hugging_face"]["token_present"] = False
    assert "WINDOWS_REQUIRED" in qa_mvp.check_preflight(report)
    assert "HF_TOKEN_MISSING" in qa_mvp.check_preflight(report)
    san = qa_mvp.sanitized_environment(report)
    assert "hf_HIDDEN" not in json.dumps(san)
    assert san["ram_gib"] == 8


def test_real_job_contract_is_exercised_without_real_models(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modules.jobs.inputs._probe_audio",
        lambda path, *, binary, timeout: (1.0, "pcm_s16le"),
    )
    # The application still uses its normal single durable JobWorker. Only
    # the heavyweight ASR instance is replaced by a deterministic fake.
    monkeypatch.setattr(
        "modules.jobs.worker.JobWorker",
        lambda jobs: RealWorker(jobs, transcriber_factory=FakeASR, poll_interval=0.01),
    )
    short = _wav(tmp_path / "private" / "short.wav")
    long = _wav(tmp_path / "private" / "long.wav")
    two = _wav(tmp_path / "private" / "two.wav")
    root = tmp_path / "ignored_qa"
    result = qa_mvp.execute_real_jobs(
        short, long, two, repeat=10, timeout_seconds=60, run_root=root,
    )
    assert result["all_passed"]
    assert result["completed_cases"] == result["required_cases"] == 13
    assert len({x["uuid"] for x in result["cases"]}) == 13
    assert all(x["metrics"]["artifact_integrity_verified"] for x in result["cases"])
    assert [x["kind"] for x in result["cases"]].count("repeat_short") == 10
    assert next(x for x in result["cases"] if x["kind"] == "two_speakers")["metrics"]["speaker_count"] >= 2
    assert all((root / "jobs" / x["uuid"]).is_dir() for x in result["cases"])


def _good_evidence():
    cases = [
        {"kind": "short", "status": "passed"},
        {"kind": "long", "status": "passed"},
        {"kind": "two_speakers", "status": "passed"},
    ] + [{"kind": "repeat_short", "status": "passed"} for _ in range(10)]
    for case in cases:
        case["metrics"] = {
            "artifact_integrity_verified": True, "txt_json_consistent": True,
            "speaker_count": 2 if case["kind"] == "two_speakers" else 1,
        }
    return {
        "status": "pipeline_passed",
        "environment": {"os": "Windows"},
        "real_inference_attempted": True,
        "all_passed": True,
        "cases": cases,
    }


def test_release_gate_fails_closed_without_actual_windows_results():
    evidence = _good_evidence()
    manual = mvp_release_gate.make_template()
    assert len(manual["tests"]) == len(mvp_release_gate.REQUIRED_MANUAL)
    assert mvp_release_gate.evaluate(evidence, manual)["status"] == "BLOCKED"
    for entry in manual["tests"].values():
        entry["status"] = "PASS"
        entry["note"] = "Tested personally on the local Windows machine."
    assert mvp_release_gate.evaluate(evidence, manual)["status"] == "READY_FOR_HUMAN_RELEASE_REVIEW"
    evidence["real_inference_attempted"] = False
    status = mvp_release_gate.evaluate(evidence, manual)
    assert status["status"] == "BLOCKED"
    assert "ACTUAL_INFERENCE_NOT_VERIFIED" in status["blockers"]
    evidence["real_inference_attempted"] = True
    evidence["environment"]["os"] = "Linux"
    assert "WINDOWS_NOT_VERIFIED" in mvp_release_gate.evaluate(evidence, manual)["blockers"]


def test_no_release_acceptance_without_two_distinct_speakers():
    evidence = _good_evidence()
    manual = mvp_release_gate.make_template()
    evidence["cases"][2]["metrics"]["speaker_count"] = 1
    assert "TWO_SPEAKER_CASE_UNVERIFIED" in mvp_release_gate.evaluate(
        evidence, manual,
    )["blockers"]


def test_gate_never_creates_tag_and_requires_explicit_manual_evidence(tmp_path):
    template = tmp_path / "manual.json"
    assert mvp_release_gate.main(["--template", str(template)]) == 0
    obj = json.loads(template.read_text(encoding="utf-8"))
    assert all(test["status"] == "NOT_RUN" for test in obj["tests"].values())
    evidence = tmp_path / "real.json"
    evidence.write_text(json.dumps(_good_evidence()), encoding="utf-8")
    assert mvp_release_gate.main(["--evaluate", str(evidence),
                                  "--manual", str(template)]) == 2

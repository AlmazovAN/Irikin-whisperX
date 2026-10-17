"""P25: opt-in on-device MVP acceptance through the *real* durable job worker.

Read-only --preflight does not load models or process any user recordings.
--run creates a new ignored local data/qa_mvp_runs/<run-id>/ directory
containing private copied audio and verified JSON/TXT bundles. It does not
start a second Gradio server, delete any user's files, or claim browser QA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.utils.environment_preflight import build_report

REQUIRED_PACKAGES = (
    "torch", "torchaudio", "whisperx", "pyannote.audio",
    "gradio", "faster-whisper", "ctranslate2",
)
TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"})
MAX_REPEAT = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_preflight(report: dict) -> list[str]:
    """Report an unmet prerequisite as BLOCKED, never as a passing smoke."""
    problems: list[str] = []
    if report.get("system", {}).get("name") != "Windows":
        problems.append("WINDOWS_REQUIRED")
    if not report.get("python", {}).get("supported_by_readme"):
        problems.append("PYTHON_UNSUPPORTED")
    for cmd in ("ffmpeg", "ffprobe"):
        if report.get(cmd, {}).get("status") != "ok":
            problems.append(cmd.upper() + "_MISSING")
    for package in REQUIRED_PACKAGES:
        if report.get("packages", {}).get(package, {}).get("status") != "installed":
            problems.append("PACKAGE_MISSING:" + package)
    if report.get("torch", {}).get("status") != "ok":
        problems.append("TORCH_UNAVAILABLE")
    if not report.get("hugging_face", {}).get("token_present"):
        problems.append("HF_TOKEN_MISSING")
    return problems


def sanitized_environment(report: dict) -> dict:
    """No usernames, absolute paths, environmental variables or token values."""
    return {
        "os": report.get("system", {}).get("name"),
        "python": report.get("python", {}).get("version"),
        "ram_gib": report.get("ram", {}).get("total_gib"),
        "torch_status": report.get("torch", {}).get("status"),
        "cuda_available": report.get("torch", {}).get("cuda_available"),
        "ffmpeg": report.get("ffmpeg", {}).get("status"),
        "ffprobe": report.get("ffprobe", {}).get("status"),
        "hf_token_present": bool(report.get("hugging_face", {}).get("token_present")),
        "packages": {
            name: report.get("packages", {}).get(name, {}).get("version")
            for name in REQUIRED_PACKAGES
        },
    }


def inspect_verified_result(service, identifier: str, *, min_speakers: int) -> dict:
    """Verify the entire atomic manifest, TXT/JSON parity and real word timing.

    Do not return transcript text, filenames or local paths.
    """
    import re as _re
    txt_file = service.get_artifact(identifier, format="txt")
    json_file = service.get_artifact(identifier, format="json")
    raw = txt_file.read_text(encoding="utf-8")
    data = json.loads(json_file.read_text(encoding="utf-8"))
    if (data.get("schema_version") != 1
            or data.get("model_size") != "small"
            or data.get("language") != "ru"
            or data.get("diarization_requested") is not True
            or data.get("diagnostic_only") is not False):
        raise ValueError("INVALID_MVP_METADATA")
    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("EMPTY_SEGMENTS")
    expected_lines: list[str] = []
    aligned = 0
    speakers: set[str] = set()
    for item in segments:
        start, end = item.get("start"), item.get("end")
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not 0 <= start <= end):
            raise ValueError("INVALID_SEGMENT_TIMES")
        label = item.get("speaker")
        if isinstance(label, str) and label.strip():
            speakers.add(label)
        words = item.get("words") or ()
        for word in words:
            a, b = word.get("start"), word.get("end")
            if (isinstance(a, (float, int)) and not isinstance(a, bool)
                    and isinstance(b, (float, int)) and not isinstance(b, bool)
                    and 0 <= a <= b):
                aligned += 1
            if isinstance(word.get("speaker"), str) and word["speaker"].strip():
                speakers.add(word["speaker"])
        text = " ".join(item.get("text", "").split())
        if not text:
            raise ValueError("EMPTY_SPOKEN_SEGMENT")
        sec = int(start)
        hours, remainder = divmod(sec, 3600)
        minutes, seconds = divmod(remainder, 60)
        speaker = f"[{label}]: " if label else ""
        expected_lines.append(f"[{hours:02d}:{minutes:02d}:{seconds:02d}] {speaker}{text}")
    if raw != "\n".join(expected_lines) + "\n":
        raise ValueError("TXT_JSON_MISMATCH")
    if aligned == 0:
        raise ValueError("ALIGNMENT_MISSING")
    if len(speakers) < min_speakers:
        raise ValueError("DIARIZATION_MISSING")
    return {
        "segments": len(segments),
        "aligned_words": aligned,
        "speaker_count": len(speakers),
        "txt_json_consistent": True,
        "artifact_integrity_verified": True,
    }


def wait_terminal(jobs, identifier: str, *, timeout_seconds: float, poll_interval: float = 0.5):
    """No artificial COMPLETED: observe the persisted SQLite status."""
    end = time.monotonic() + timeout_seconds
    while time.monotonic() < end:
        ticket = jobs.get_status(identifier)
        if ticket is None:
            raise RuntimeError("JOB_VANISHED")
        if ticket.status in TERMINAL:
            return ticket
        time.sleep(poll_interval)
    raise TimeoutError("JOB_TIMEOUT")


def execute_real_jobs(
    short: Path, long: Path, two_speakers: Path,
    *, repeat: int, timeout_seconds: int, run_root: Path,
) -> dict:
    """Run one real JobWorker through SQLite and atomic results, FIFO.

    This is not a browser test. Input files are always duplicated into a
    separate ignored QA directory; nothing is auto-deleted on error.
    """
    from modules.jobs.inputs import InputSource
    from modules.jobs.service import JobService
    from modules.jobs.worker import JobWorker

    selections = [
        ("short", short, 1),
        ("long", long, 1),
        ("two_speakers", two_speakers, 2),
    ] + [("repeat_short", short, 1) for _ in range(repeat)]
    paths = [audio.expanduser().resolve(strict=True) for _, audio, _ in selections]
    approved = tuple(dict.fromkeys(p.parent for p in paths))
    jobs = JobService(storage_root=run_root / "jobs", local_roots=approved)
    worker = JobWorker(jobs)
    cases: list[dict] = []
    worker.start()
    timed_out = False
    try:
        for case_index, ((kind, _original, min_speakers), audio) in enumerate(
            zip(selections, paths), start=1
        ):
            case = {"index": case_index, "kind": kind, "status": "not_run",
                    "actual_inference_attempted": False}
            cases.append(case)
            try:
                created_at = time.monotonic()
                ticket = jobs.submit(InputSource.local_file(audio))
                case["uuid"] = ticket.identifier
                worker.wake()
                case["actual_inference_attempted"] = True
                final = wait_terminal(jobs, ticket.identifier, timeout_seconds=timeout_seconds)
                case["status"] = final.status.lower()
                case["stage"] = final.stage
                case["attempt"] = final.attempt
                case["error_code"] = final.error_code
                case["elapsed_seconds"] = round(time.monotonic() - created_at, 2)
                if final.status == "COMPLETED":
                    case["metrics"] = inspect_verified_result(
                        jobs, ticket.identifier, min_speakers=min_speakers,
                    )
                    case["status"] = "passed"
                else:
                    break
            except TimeoutError:
                case["status"] = "timed_out"
                case["error_code"] = "JOB_TIMEOUT"
                timed_out = True
                # Ask worker to stop cooperatively, never kill the model or
                # requeue a half-finished job and report false success.
                if case.get("uuid"):
                    jobs.cancel(case["uuid"])
                break
            except Exception as exc:
                case["status"] = "failed"
                case["error_code"] = (
                    str(exc) if str(exc) in {
                        "INVALID_MVP_METADATA", "EMPTY_SEGMENTS",
                        "INVALID_SEGMENT_TIMES", "EMPTY_SPOKEN_SEGMENT",
                        "TXT_JSON_MISMATCH", "ALIGNMENT_MISSING",
                        "DIARIZATION_MISSING", "JOB_VANISHED",
                    } else type(exc).__name__
                )
                break
    finally:
        worker.stop(wait=not timed_out, timeout=15 if not timed_out else None)
    seen = [case.get("uuid") for case in cases]
    all_passed = (
        len(cases) == len(selections)
        and len(set(seen)) == len(seen)
        and all(case["status"] == "passed" for case in cases)
    )
    # Verify earlier result files still exist after all later runs.
    if all_passed:
        try:
            for case in cases:
                jobs.get_artifact(case["uuid"], format="json")
                jobs.get_artifact(case["uuid"], format="txt")
        except Exception:
            all_passed = False
    return {"cases": cases, "all_passed": all_passed,
            "completed_cases": sum(c["status"] == "passed" for c in cases),
            "required_cases": len(selections)}


def safe_write(report: dict, directory: Path) -> Path:
    """Private audit evidence; no media/text; never overwrite prior run."""
    directory.mkdir(parents=True, exist_ok=True)
    name = "qa-mvp-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    target = directory / (name + ".json")
    temp = directory / (name + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as out:
            json.dump(report, out, ensure_ascii=False, indent=2)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        temp.rename(target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP P25 Windows real worker acceptance")
    tests = parser.add_mutually_exclusive_group(required=True)
    tests.add_argument("--preflight", action="store_true",
                       help="Read-only environment check; never start WhisperX")
    tests.add_argument("--run", action="store_true",
                       help="Explicitly run real models and persist private copies of audio")
    parser.add_argument("--short", type=Path)
    parser.add_argument("--long", type=Path)
    parser.add_argument("--two-speakers", type=Path)
    parser.add_argument("--repeat", type=int, default=MAX_REPEAT)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args(argv)
    if args.repeat != MAX_REPEAT:
        parser.error("MVP release gate requires exactly 10 repeat runs")
    if not 60 <= args.timeout_seconds <= 86400:
        parser.error("--timeout-seconds must be between 60 and 86400")
    if args.run and any(getattr(args, label) is None for label in (
        "short", "long", "two_speakers",
    )):
        parser.error("--run requires --short, --long and --two-speakers")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (ROOT / ".env").is_file():
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    context = build_report(check_hf=False, skip_torch=False)
    blockers = check_preflight(context)
    output: dict = {
        "schema_version": 1,
        "procedure": "P25 local end-to-end durable job pipeline",
        "observed_utc": utc_now(),
        "status": "blocked" if blockers else "preflight_only",
        "real_inference_attempted": False,
        "browser_qa_performed": False,
        "release_accepted": False,
        "environment": sanitized_environment(context),
        "blocker_codes": blockers,
    }
    code = 2 if blockers else 0
    if args.run and not blockers:
        run_root = ROOT / "data" / "qa_mvp_runs" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-" + uuid4().hex[:8]
        )
        output["real_inference_attempted"] = True
        output["private_result_directory_created"] = True
        try:
            result = execute_real_jobs(
                args.short, args.long, args.two_speakers,
                repeat=args.repeat, timeout_seconds=args.timeout_seconds,
                run_root=run_root,
            )
            output.update(result)
            output["status"] = "pipeline_passed" if result["all_passed"] else "failed"
            code = 0 if result["all_passed"] else 1
        except Exception as exc:
            output["status"] = "failed"
            output["failure_class"] = type(exc).__name__
            code = 1
    elif args.run:
        output["status"] = "blocked"
    location = safe_write(output, ROOT / "logs")
    print("P25 pipeline QA:", output["status"].upper())
    print("Browser Windows QA:", "NOT_RUN (independent manual protocol)")
    print("MVP release gate:", "BLOCKED until documented on-device browser evidence")
    print("Private report:", location)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

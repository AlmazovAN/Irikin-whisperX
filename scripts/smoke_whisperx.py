"""P06 manual Windows smoke test of the *existing* WhisperX pipeline.

No UI, no second inference engine and no checked-in audio. A real pass requires
an explicitly supplied local Russian WAV/MP3 and actual model execution.
This CLI never installs dependencies or transcribes in --diagnose-only mode.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.utils.environment_preflight import build_report

MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_DURATION_SECONDS = 90.0
REQUIRED_PACKAGES = (
    "torch", "torchaudio", "gradio", "whisperx", "pyannote.audio",
    "faster-whisper", "ctranslate2", "numpy",
)
EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 2
EXIT_PARTIAL = 3


class SmokeBlocked(RuntimeError):
    """An unmet prerequisite, not an inference failure."""


class SmokeFailed(RuntimeError):
    """Inference returned invalid/incomplete output, not a success."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually verify local Russian WhisperX Small VAD/ASR/alignment/diarization"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audio", type=Path, help="Private local 5–30 sec Russian WAV/MP3 (never uploaded)")
    mode.add_argument("--diagnose-only", action="store_true",
                      help="Check prerequisites; never load models or transcribe")
    parser.add_argument("--without-diarization", action="store_true",
                        help="Development diagnostic only; even if ASR works this is PARTIAL, not PASS")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="CPU by default for AMD iGPU / 8 GiB RAM")
    parser.add_argument("--compute-type", choices=("int8", "float32", "float16"), default="int8")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="WhisperX ASR batch size 1–8; default 1 for low-memory CPU")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 8:
        parser.error("--batch-size must be 1 through 8")
    if args.device == "cpu" and args.compute_type == "float16":
        parser.error("float16 on CPU is unsupported; use int8 or float32")
    return args


def prerequisites(report: dict, *, device: str, require_diarization: bool) -> list[str]:
    """Use actual observations; never infer a working model from package metadata."""
    failures = list(report.get("assessment", {}).get("blockers", []))
    if report.get("system", {}).get("name") != "Windows":
        failures.append("This smoke acceptance targets Windows; other OS does not validate it")
    packages = report.get("packages", {})
    for package in REQUIRED_PACKAGES:
        if packages.get(package, {}).get("status") != "installed":
            failures.append(f"Missing or broken package: {package}")
    torch_info = report.get("torch", {})
    if torch_info.get("status") != "ok":
        failures.append("PyTorch device probe has not succeeded")
    if device == "cuda" and not torch_info.get("cuda_available", False):
        failures.append("CUDA device is unavailable; use --device cpu on AMD iGPU")
    if require_diarization and not report.get("hugging_face", {}).get("token_present", False):
        failures.append("Hugging Face token absent; set HF_TOKEN in private .env")
    return list(dict.fromkeys(failures))


def inspect_audio(path: Path) -> dict:
    """No decoding/model loading; constrain size, duration and accepted format."""
    candidate = path.expanduser().resolve()
    if not candidate.is_file():
        raise SmokeBlocked("Audio file does not exist; provide a local WAV/MP3")
    if candidate.suffix.lower() not in (".wav", ".mp3"):
        raise SmokeBlocked("Only WAV and MP3 inputs are supported by the P06 acceptance")
    size = candidate.stat().st_size
    if size <= 0 or size > MAX_AUDIO_BYTES:
        raise SmokeBlocked("Audio must be nonempty and no larger than 25 MiB")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise SmokeBlocked("ffprobe is unavailable")
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(candidate)],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if proc.returncode != 0:
            raise SmokeBlocked("ffprobe could not decode the input metadata")
        duration = float(json.loads(proc.stdout)["format"]["duration"])
    except SmokeBlocked:
        raise
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as exc:
        raise SmokeBlocked(f"Could not inspect audio: {type(exc).__name__}") from exc
    if not (0.1 <= duration <= MAX_DURATION_SECONDS):
        raise SmokeBlocked("Provide a short audio recording of at most 90 seconds")
    # Only nonidentifying metadata is recorded; audio path and spoken text are not.
    return {"format": candidate.suffix.lower(), "size_bytes": size,
            "duration_seconds": round(duration, 2)}


def evaluate_segments(segments: list, *, require_diarization: bool) -> dict:
    """Demand actual ASR text, aligned word timestamps and speaker labels."""
    if not segments:
        raise SmokeFailed("Pipeline returned zero segments")
    spoken = [segment for segment in segments if str(getattr(segment, "text", "") or "").strip()]
    if not spoken:
        raise SmokeFailed("No recognized text; silent/invalid audio or model failure")
    words = [word for segment in spoken for word in (getattr(segment, "words", None) or [])]
    aligned = [word for word in words
               if getattr(word, "start", None) is not None
               and getattr(word, "end", None) is not None]
    if not aligned:
        raise SmokeFailed("WhisperX alignment produced no word timestamps")
    speaker_labels = {
        label for segment in spoken
        for label in (
            [getattr(segment, "speaker", None)]
            + [getattr(word, "speaker", None) for word in (getattr(segment, "words", None) or [])]
        )
        if isinstance(label, str) and label.strip()
    }
    if require_diarization and not speaker_labels:
        raise SmokeFailed("Diarization did not assign speaker labels")
    return {
        "segments_with_text": len(spoken),
        "aligned_words": len(aligned),
        "speaker_count": len(speaker_labels),
        "asr_verified": True,
        "alignment_verified": True,
        "diarization_verified": require_diarization and bool(speaker_labels),
        "result_contains_spoken_text": False,
    }


def execute_pipeline(audio: Path, args: argparse.Namespace) -> dict:
    """One inference instance; call the existing BaseTranscriptionPipeline.run."""
    import gradio as gr
    import torch
    from modules.mvp.preset import MVP_PRESET, build_mvp_pipeline_params
    from modules.whisper.whisperx_inference import WhisperXInference

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SmokeBlocked("CUDA was requested but no working CUDA device is detected")
    require_diarization = not args.without_diarization
    params = build_mvp_pipeline_params(
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        diarization_enabled=require_diarization,
    )
    started = time.monotonic()
    pipeline = WhisperXInference(
        model_dir=str(ROOT / "models" / "Whisper"),
        diarization_model_dir=str(ROOT / "models" / "Diarization"),
        output_dir=str(ROOT / "outputs"),
    )
    pipeline.device = args.device
    try:
        segments, elapsed = pipeline.run(
            str(audio), gr.Progress(), MVP_PRESET.output_format, False, None,
            *params.to_list(), persist_settings=False, strict_vad=True
        )
        validated = evaluate_segments(segments, require_diarization=require_diarization)
        validated["elapsed_pipeline_seconds"] = round(float(elapsed), 2)
        validated["elapsed_total_seconds"] = round(time.monotonic() - started, 2)
        validated["configuration"] = {
            "engine": "existing WhisperXInference.run",
            "model": MVP_PRESET.model_size,
            "language": MVP_PRESET.language, "task": MVP_PRESET.task,
            "device_requested": args.device, "compute_type_requested": args.compute_type,
            "batch_size": args.batch_size, "vad_enabled": MVP_PRESET.vad_enabled,
            "alignment_enabled": MVP_PRESET.alignment_enabled,
            "diarization_requested": require_diarization,
        }
        if args.device == "cuda" and torch.cuda.is_available():
            validated["cuda_peak_allocated_mib"] = round(
                torch.cuda.max_memory_allocated() / (1024 ** 2), 1
            )
        return validated
    finally:
        # No second engine and no concurrent model execution.
        # Offload is best effort; let a failing cleanup propagate to the caller.
        pipeline.offload()
        if getattr(pipeline, "whisperx_wrapper", None) is not None:
            pipeline.whisperx_wrapper.offload_diarizer()


def sanitized_environment(report: dict) -> dict:
    """Capture only reproducibility metadata, not local paths or HF credentials."""
    return {
        "os": report.get("system", {}).get("name"),
        "os_release": report.get("system", {}).get("release"),
        "python": report.get("python", {}).get("version"),
        "ram_gib": report.get("ram", {}).get("total_gib"),
        "free_disk_gib": report.get("disk", {}).get("free_gib"),
        "ffmpeg": report.get("ffmpeg", {}).get("status"),
        "ffprobe": report.get("ffprobe", {}).get("status"),
        "gpus": [
            item.get("name") for item in report.get("windows_gpus", {}).get("adapters", [])
        ],
        "nvidia_vram_mib": [
            item.get("vram_mib") for item in report.get("nvidia_smi", {}).get("devices", [])
        ],
        "torch": {
            "status": report.get("torch", {}).get("status"),
            "cuda_available": report.get("torch", {}).get("cuda_available"),
            "torch_cuda_build": report.get("torch", {}).get("torch_cuda_build"),
        },
        "package_versions": {
            name: info.get("version") if info.get("status") == "installed" else None
            for name, info in report.get("packages", {}).items()
        },
        "hf_token_present": bool(report.get("hugging_face", {}).get("token_present")),
        "hf_gated_model_access_tested": False,
    }


def save_report(payload: dict, *, root: Path | None = None) -> Path:
    base = root if root is not None else ROOT
    logs = base / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    name = "smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    path = logs / (name + ".json")
    temporary = logs / (name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = {
        "schema_version": 1,
        "test": "P06 manual Russian WhisperX Small pipeline",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked",
        "actual_inference_performed": False,
        "inference_attempted": False,
        "diarization_required": not args.without_diarization,
        "stages": {
            "vad": "not_run", "asr": "not_run",
            "alignment": "not_run", "diarization": "not_run",
        },
    }
    code = EXIT_BLOCKED
    try:
        # dotenv loading is opt-in by the existence of a *local* file;
        # never print/serialize the token value or the .env file's contents.
        if (ROOT / ".env").is_file():
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
        observations = build_report(check_hf=False, skip_torch=False)
        report["environment"] = sanitized_environment(observations)
        failures = prerequisites(
            observations, device=args.device,
            require_diarization=not args.without_diarization
        )
        if failures:
            report["blockers"] = failures
        elif args.diagnose_only:
            report["status"] = "ready_for_manual_audio"
            report["note"] = "Environment metadata is present; no audio/model test performed"
            code = EXIT_PASSED
        else:
            report["audio"] = inspect_audio(args.audio)
            report["stages"] = {
                "vad": "enabled_not_independently_verified",
                "asr": "requested", "alignment": "requested",
                "diarization": "requested" if not args.without_diarization else "skipped",
            }
            # No real-model imports until all read-only checks succeeded.
            report["inference_attempted"] = True
            metrics = execute_pipeline(args.audio.expanduser().resolve(), args)
            report["actual_inference_performed"] = True
            report["metrics"] = metrics
            report["stages"].update({
                "asr": "verified", "alignment": "verified",
                "diarization": "verified" if metrics["diarization_verified"] else "skipped",
            })
            report["status"] = "partial" if args.without_diarization else "passed"
            code = EXIT_PARTIAL if args.without_diarization else EXIT_PASSED
    except SmokeBlocked as exc:
        report["status"] = "blocked"
        report["blockers"] = [str(exc)]
    except SmokeFailed as exc:
        report["status"] = "failed"
        report["error_class"] = "SmokeFailed"
        report["failure_detail"] = str(exc)
        code = EXIT_FAILED
    except Exception as exc:
        report["status"] = "failed"
        report["error_class"] = type(exc).__name__
        report["failure_detail"] = "Check local console logs privately; exception message withheld to protect credentials"
        code = EXIT_FAILED
    try:
        destination = save_report(report)
    except OSError as exc:
        print(f"ERROR: diagnostic report could not be written ({type(exc).__name__})", file=sys.stderr)
        return EXIT_FAILED
    print(f"P06 RESULT: {report['status'].upper()} (real inference: {report['actual_inference_performed']})")
    if report.get("blockers"):
        for item in report["blockers"]:
            print("BLOCKER:", item)
    if report.get("error_class"):
        print("FAILURE TYPE:", report["error_class"])
    print("Report:", destination)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

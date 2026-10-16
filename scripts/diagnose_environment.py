"""Command-line Windows preflight for Irikin-whisperX (stdlib + optional torch)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.utils.environment_preflight import build_report, render_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only local WhisperX environment diagnostics")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the text summary")
    parser.add_argument("--output", type=Path, help="Also save the report as a JSON file")
    parser.add_argument("--check-hf", action="store_true", help="Opt-in HTTPS token authentication check on huggingface.co (never verifies gated model access)")
    parser.add_argument("--skip-torch", action="store_true", help="Do not import installed torch")
    args = parser.parse_args(argv)
    report = build_report(check_hf=args.check_hf, skip_torch=args.skip_torch)
    result_json = json.dumps(report, ensure_ascii=False, indent=2)
    print(result_json if args.json else render_text(report))
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result_json + "\n", encoding="utf-8")
            print("JSON report saved.", file=sys.stderr)
        except OSError as exc:
            print(f"Failed to write report: {type(exc).__name__}", file=sys.stderr)
            return 1
    return 0  # Diagnostics completed; blockers are reported, not hidden in exit code.


if __name__ == "__main__":
    raise SystemExit(main())

"""P25 human-confirmed release gate. Never runs models or creates Git tags.

A successful code CI is not Windows/browser acceptance. This script requires
both an actual Windows full-pipeline report and explicit human observations;
it can recommend review, not certify the authenticity of user-entered notes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_MANUAL = (
    "windows_install_and_start",
    "browser_ru_short_completed",
    "browser_long_audio_completed",
    "browser_two_speakers_confirmed",
    "tab_minimized_survives",
    "tab_closed_survives",
    "tab_reload_reconnects",
    "server_restart_recovery",
    "offline_after_models_cached",
    "manual_txt_download",
    "automatic_download_or_manual_fallback",
    "history_and_preview",
    "open_folder_windows",
    "safe_delete_confirmation",
    "failure_does_not_show_completed",
    "no_duplicate_worker",
)


def make_template() -> dict:
    return {
        "schema_version": 1,
        "purpose": "P25 manual Windows browser QA; fill only after observing results",
        "environment": "Windows 11 / local CPU profile",
        "tests": {key: {"status": "NOT_RUN", "note": ""} for key in REQUIRED_MANUAL},
    }


def evaluate(real_report: dict, manual_report: dict) -> dict:
    """Fail closed; never accept preflight-only or mock inference evidence."""
    blockers: list[str] = []
    if real_report.get("status") != "pipeline_passed":
        blockers.append("REAL_PIPELINE_NOT_PASSED")
    if real_report.get("environment", {}).get("os") != "Windows":
        blockers.append("WINDOWS_NOT_VERIFIED")
    if real_report.get("real_inference_attempted") is not True:
        blockers.append("ACTUAL_INFERENCE_NOT_VERIFIED")
    if real_report.get("all_passed") is not True:
        blockers.append("JOB_CASES_INCOMPLETE")
    cases = real_report.get("cases")
    if (not isinstance(cases, list) or len(cases) != 13
            or sum(x.get("kind") == "repeat_short" for x in cases if isinstance(x, dict)) != 10
            or sum(x.get("kind") == "two_speakers" for x in cases if isinstance(x, dict)) != 1
            or sum(x.get("kind") == "long" for x in cases if isinstance(x, dict)) != 1
            or any(not isinstance(x, dict) or x.get("status") != "passed"
                   or not x.get("metrics", {}).get("artifact_integrity_verified")
                   or not x.get("metrics", {}).get("txt_json_consistent")
                   for x in cases)):
        blockers.append("REQUIRED_CASE_EVIDENCE_INCOMPLETE")
    else:
        two = next(x for x in cases if x["kind"] == "two_speakers")
        if two["metrics"].get("speaker_count", 0) < 2:
            blockers.append("TWO_SPEAKER_CASE_UNVERIFIED")

    entries = manual_report.get("tests", {})
    if not isinstance(entries, dict):
        entries = {}
    for key in REQUIRED_MANUAL:
        entry = entries.get(key)
        if (not isinstance(entry, dict)
                or entry.get("status") != "PASS"
                or not isinstance(entry.get("note"), str)
                or not entry["note"].strip()):
            blockers.append("MANUAL_NOT_PASSED:" + key)
    return {
        "status": "READY_FOR_HUMAN_RELEASE_REVIEW" if not blockers else "BLOCKED",
        "blockers": blockers,
        "tag_created": False,
        "note": "Local evidence is self-reported; review actual PC/browser results and CI before tagging.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conservative MVP P25 release gate")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--template", type=Path,
                         help="Create manual checklist JSON with all statuses NOT_RUN")
    actions.add_argument("--evaluate", type=Path,
                         help="Read private full-pipeline QA JSON")
    parser.add_argument("--manual", type=Path, help="Filled manual checklist for --evaluate")
    args = parser.parse_args(argv)
    if args.template is not None:
        if args.template.exists():
            parser.error("Template file already exists; refusing overwrite")
        args.template.parent.mkdir(parents=True, exist_ok=True)
        with args.template.open("x", encoding="utf-8") as stream:
            json.dump(make_template(), stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print("Template created; all tests are NOT_RUN.")
        return 0
    if args.manual is None:
        parser.error("--evaluate requires --manual")
    try:
        actual = json.loads(args.evaluate.read_text(encoding="utf-8"))
        manual = json.loads(args.manual.read_text(encoding="utf-8"))
        result = evaluate(actual, manual)
    except (OSError, ValueError, TypeError, AttributeError):
        print("MVP RELEASE: BLOCKED (cannot read or validate local evidence)")
        return 2
    print("MVP RELEASE:", result["status"])
    for blocker in result["blockers"]:
        print("BLOCKER:", blocker)
    print("No GitHub tag or release has been created.")
    return 0 if result["status"] == "READY_FOR_HUMAN_RELEASE_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())

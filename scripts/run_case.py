#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIG = "configs/evaluation/final_pipeline_eval.yaml"
DEFAULT_EVALUATE_SCRIPT = "scripts/evaluate_case.py"
DEFAULT_EXTRACT_SCRIPT = "scripts/extract_attack_steps.py"


POWERSHELL_STOP_OLLAMA = 'Stop-Process -Name "ollama*" -Force -ErrorAction SilentlyContinue'
POWERSHELL_START_OLLAMA = 'Start-Process "$env:LOCALAPPDATA\\Programs\\Ollama\\ollama app.exe"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a case one attack step at a time and restart Ollama after every step."
    )
    parser.add_argument("--case", required=True, help="Case ID to run, for example: goofy")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Evaluation config path. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--evaluate-script",
        default=DEFAULT_EVALUATE_SCRIPT,
        help=f"Single-step evaluator script. Default: {DEFAULT_EVALUATE_SCRIPT}",
    )
    parser.add_argument(
        "--extract-script",
        default=DEFAULT_EXTRACT_SCRIPT,
        help=f"Attack-step extraction script. Default: {DEFAULT_EXTRACT_SCRIPT}",
    )
    parser.add_argument(
        "--steps-file",
        default=None,
        help=(
            "Optional CSV/JSONL file containing attack_step_id values. If omitted, "
            "the script calls extract_attack_steps.py to create a temporary CSV."
        ),
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help=(
            "Directory used for temporary step-index files and run logs. "
            "Default: case_outputs/<case_id>."
        ),
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Optional attack_step_id or numeric step_index from which to start.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of attack steps to run.",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Pass --recover to evaluate_case2.py for each attack step.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next attack step if evaluate_case2.py exits with a non-zero code.",
    )
    parser.add_argument(
        "--restart-wait",
        type=float,
        default=10.0,
        help="Seconds to wait after starting Ollama. Default: 10.",
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Do not restart Ollama. Useful for dry testing the orchestration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without running evaluation or restarting Ollama.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    work_dir = resolve_path(args.work_dir) if args.work_dir else REPO_ROOT / "case_outputs" / args.case
    work_dir.mkdir(parents=True, exist_ok=True)

    log_path = work_dir / "run_case_log.jsonl"

    step_rows = load_or_extract_attack_steps(args, work_dir)
    step_rows = select_requested_slice(step_rows, start_at=args.start_at, limit=args.limit)

    if not step_rows:
        raise RuntimeError(f"No attack steps selected for case {args.case}.")

    print(f"Selected {len(step_rows)} attack step(s) for case '{args.case}'.")
    print(f"Run log: {log_path}")

    for index, row in enumerate(step_rows, start=1):
        attack_step_id = str(row["attack_step_id"])
        step_index = row.get("step_index", "")
        label = row.get("attack_step_label", "")

        print("\n" + "=" * 88)
        print(f"[{index}/{len(step_rows)}] attack_step_id={attack_step_id}")
        if step_index != "":
            print(f"step_index={step_index}")
        if label:
            print(f"label={label}")
        print("=" * 88)

        command = build_evaluate_command(args, attack_step_id)
        print("Command:", shell_join(command))

        started_at = utc_now()
        return_code: int | None = None
        status = "planned" if args.dry_run else "started"
        error_message = ""

        try:
            if not args.dry_run:
                completed = subprocess.run(command, cwd=REPO_ROOT)
                return_code = int(completed.returncode)
                status = "completed" if return_code == 0 else "failed"
                if return_code != 0:
                    error_message = f"evaluate_case2.py exited with code {return_code}"
                    print(f"ERROR: {error_message}")

            append_jsonl(
                log_path,
                {
                    "timestamp_utc": utc_now(),
                    "case_id": args.case,
                    "attack_step_id": attack_step_id,
                    "step_index": step_index,
                    "attack_step_label": label,
                    "status": status,
                    "return_code": return_code,
                    "started_at_utc": started_at,
                    "finished_at_utc": utc_now(),
                    "command": command,
                    "error": error_message,
                },
            )

            if return_code not in (None, 0) and not args.continue_on_error:
                print("Stopping because --continue-on-error was not set.")
                return return_code

        finally:
            # The requested behavior is to restart Ollama after each step. We do
            # it in finally so the restart also happens after a failed step.
            if not args.no_restart and not args.dry_run:
                restart_ollama(wait_seconds=args.restart_wait, log_path=log_path, case_id=args.case, attack_step_id=attack_step_id)
            elif args.dry_run:
                print("Dry run: would restart Ollama here.")

    print("\nCompleted selected attack steps.")
    return 0


def load_or_extract_attack_steps(args: argparse.Namespace, work_dir: Path) -> list[dict[str, Any]]:
    """Load attack steps from --steps-file or create a fresh index with extract_attack_steps.py."""
    if args.steps_file:
        steps_path = resolve_path(args.steps_file)
        print(f"Loading attack steps from: {steps_path}")
        return read_step_index(steps_path)

    steps_path = work_dir / "attack_steps_index_for_run_case.csv"
    extract_script = resolve_path(args.extract_script)
    config_path = resolve_path(args.config)

    command = [
        sys.executable,
        str(extract_script),
        "--case",
        args.case,
        "--config",
        str(config_path),
        "--format",
        "csv",
        "--output",
        str(steps_path),
        "--persist-jsonl",
    ]

    print("Extracting attack-step index:", shell_join(command))
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"extract_attack_steps.py failed with exit code {completed.returncode}")

    return read_step_index(steps_path)


def read_step_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Attack-step index not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rows.append(normalize_step_row(record))
        return validate_step_rows(rows, path)

    # Default to CSV because extract_attack_steps.py writes CSV by default.
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [normalize_step_row(dict(row)) for row in reader]
    return validate_step_rows(rows, path)


def normalize_step_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a row from either the compact CSV index or raw JSONL records."""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}

    attack_step_id = row.get("attack_step_id") or row.get("id")
    step_index = row.get("step_index") or metadata.get("step_index") or ""
    attack_step_label = row.get("attack_step_label") or row.get("original_label") or row.get("label") or ""

    return {
        **row,
        "attack_step_id": str(attack_step_id or "").strip(),
        "step_index": str(step_index).strip() if step_index is not None else "",
        "attack_step_label": str(attack_step_label or "").strip(),
    }


def validate_step_rows(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError(f"No attack steps found in {path}")

    seen: set[str] = set()
    duplicates: set[str] = set()
    missing = 0

    for row in rows:
        step_id = str(row.get("attack_step_id", "")).strip()
        if not step_id:
            missing += 1
            continue
        if step_id in seen:
            duplicates.add(step_id)
        seen.add(step_id)

    if missing:
        raise ValueError(f"{path} contains {missing} row(s) without attack_step_id")
    if duplicates:
        raise ValueError(f"{path} contains duplicate attack_step_id values: {sorted(duplicates)}")

    return rows


def select_requested_slice(rows: list[dict[str, Any]], *, start_at: str | None, limit: int | None) -> list[dict[str, Any]]:
    selected = list(rows)

    if start_at:
        start_position = find_start_position(selected, start_at)
        selected = selected[start_position:]

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        selected = selected[:limit]

    return selected


def find_start_position(rows: list[dict[str, Any]], start_at: str) -> int:
    requested = str(start_at).strip()
    requested_lower = requested.lower()

    for i, row in enumerate(rows):
        attack_step_id = str(row.get("attack_step_id", "")).strip()
        if attack_step_id == requested or attack_step_id.lower() == requested_lower:
            return i

    # Also allow numeric step_index because the attack-step index exposes it.
    for i, row in enumerate(rows):
        step_index = str(row.get("step_index", "")).strip()
        if step_index == requested:
            return i

    available = "\n".join(f"- {row.get('step_index', '')}: {row['attack_step_id']}" for row in rows[:50])
    raise ValueError(f"--start-at value not found: {requested}\nAvailable first rows:\n{available}")


def build_evaluate_command(args: argparse.Namespace, attack_step_id: str) -> list[str]:
    command = [
        sys.executable,
        str(resolve_path(args.evaluate_script)),
        "--case",
        args.case,
        "--config",
        str(resolve_path(args.config)),
        "--as",
        attack_step_id,
    ]

    if args.recover:
        command.append("--recover")

    return command


def restart_ollama(*, wait_seconds: float, log_path: Path, case_id: str, attack_step_id: str) -> None:
    """Restart the Windows Ollama desktop app using PowerShell."""
    print("Restarting Ollama...")

    stop_result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", POWERSHELL_STOP_OLLAMA],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    # Give Windows a moment to release the model runner and GPU memory.
    time.sleep(2.0)

    start_result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", POWERSHELL_START_OLLAMA],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    if wait_seconds > 0:
        print(f"Waiting {wait_seconds:.1f}s for Ollama to restart...")
        time.sleep(wait_seconds)

    append_jsonl(
        log_path,
        {
            "timestamp_utc": utc_now(),
            "case_id": case_id,
            "attack_step_id": attack_step_id,
            "event": "ollama_restart",
            "stop_return_code": stop_result.returncode,
            "start_return_code": start_result.returncode,
            "stop_stderr": (stop_result.stderr or "").strip(),
            "start_stderr": (start_result.stderr or "").strip(),
            "restart_wait_seconds": wait_seconds,
        },
    )

    if stop_result.returncode != 0:
        print(f"Warning: Stop-Process returned {stop_result.returncode}: {stop_result.stderr.strip()}")
    if start_result.returncode != 0:
        print(f"Warning: Start-Process returned {start_result.returncode}: {start_result.stderr.strip()}")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def shell_join(command: list[str]) -> str:
    """Human-readable command string for logs."""
    return " ".join(quote_for_display(part) for part in command)


def quote_for_display(part: str) -> str:
    text = str(part)
    if any(ch.isspace() for ch in text) or '"' in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
Validate structured experiment run outputs.

This script checks that run directories contain expected JSONL files and that
records expose the fields needed for annotation, scoring, and reproducibility.
It writes a CSV report with ERROR/WARNING/INFO rows.

Example:
    python scripts/validate_run_outputs.py --runs outputs/runs --out outputs/validation/run_output_validation.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.common import write_csv_rows
from src.io_utils import read_jsonl


REQUIRED_CHUNK_FIELDS = ["chunk_id", "document_id", "case_id", "chunking_strategy", "text", "source_path"]
REQUIRED_ATTACK_STEP_FIELDS = ["attack_step_id", "case_id", "original_label", "attack_tactic", "input_text"]
REQUIRED_RETRIEVAL_FIELDS = ["run_id", "case_id", "attack_step_id", "retriever", "results"]
REQUIRED_CONDITION_FIELDS = [
    "condition_id",
    "run_id",
    "case_id",
    "attack_step_id",
    "condition_type",
    "category",
    "statement",
    "evidence_chunk_ids",
    "source_documents",
    "support_level",
    "directness",
    "status",
]
REQUIRED_DESCRIPTION_FIELDS = [
    "run_id",
    "case_id",
    "attack_step_id",
    "description",
    "evidence_chunk_ids",
    "source_documents",
    "support_level",
]

VALID_CONDITION_TYPES = {"pre", "post"}
VALID_CATEGORIES = {
    "environment",
    "privilege",
    "dependency",
    "filesystem",
    "registry",
    "process",
    "network",
    "identity",
    "log_telemetry",
    "attacker_capability",
    "other",
}
VALID_SUPPORT = {"explicit", "inferred", "unsupported"}
VALID_DIRECTNESS = {"direct", "indirect", "campaign_level"}
VALID_STATUS = {"accepted", "rejected", "unknown"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate experiment run outputs.")
    parser.add_argument("--runs", default="outputs/runs", help="Directory containing run output folders.")
    parser.add_argument("--run", action="append", help="Specific run_id to validate. Can be repeated.")
    parser.add_argument("--out", default="outputs/validation/run_output_validation.csv", help="Validation report CSV path.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with code 1 if any ERROR is found.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = _discover_run_dirs(Path(args.runs), args.run)
    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        rows.extend(validate_run_dir(run_dir))

    write_csv_rows(args.out, rows)
    num_errors = sum(1 for row in rows if row.get("level") == "ERROR")
    num_warnings = sum(1 for row in rows if row.get("level") == "WARNING")
    print(f"Validated runs: {len(run_dirs)}")
    print(f"Errors: {num_errors}; Warnings: {num_warnings}")
    print(f"Report: {args.out}")

    if args.fail_on_error and num_errors:
        raise SystemExit(1)


def validate_run_dir(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_id = run_dir.name

    if not (run_dir / "config.json").exists():
        rows.append(_row(run_id, "ERROR", "config", "Missing config.json", str(run_dir / "config.json")))
        return rows

    config = _read_json(run_dir / "config.json")
    generation_enabled = bool(config.get("generation", {}).get("enabled", True))
    retrieval_enabled = bool(config.get("retrieval", {}).get("enabled", True))
    verification_enabled = bool(config.get("verification", {}).get("enabled", True))

    rows.extend(_validate_jsonl_file(run_id, run_dir, "chunks.jsonl", REQUIRED_CHUNK_FIELDS, required=True))
    rows.extend(_validate_jsonl_file(run_id, run_dir, "attack_steps.jsonl", REQUIRED_ATTACK_STEP_FIELDS, required=True))
    rows.extend(_validate_jsonl_file(run_id, run_dir, "retrieval_results.jsonl", REQUIRED_RETRIEVAL_FIELDS, required=True))
    rows.extend(_validate_jsonl_file(run_id, run_dir, "runtime_metrics.jsonl", ["run_id", "case_id", "stage", "elapsed_ms"], required=True))

    if generation_enabled:
        rows.extend(_validate_jsonl_file(run_id, run_dir, "generated_descriptions.jsonl", REQUIRED_DESCRIPTION_FIELDS, required=True))
        rows.extend(_validate_jsonl_file(run_id, run_dir, "generated_conditions.jsonl", REQUIRED_CONDITION_FIELDS, required=True))
        rows.extend(_validate_jsonl_file(run_id, run_dir, "verified_conditions.jsonl", REQUIRED_CONDITION_FIELDS, required=True))
        rows.extend(_validate_jsonl_file(run_id, run_dir, "final_conditions.jsonl", REQUIRED_CONDITION_FIELDS, required=False))
    else:
        rows.append(_row(run_id, "INFO", "generation", "Generation disabled; generated files may be absent", ""))

    chunks = read_jsonl(run_dir / "chunks.jsonl")
    chunk_ids = {str(row.get("chunk_id", "")) for row in chunks if row.get("chunk_id")}
    rows.extend(_validate_unique_ids(run_id, "chunks", chunks, "chunk_id"))
    rows.extend(_validate_unique_ids(run_id, "attack_steps", read_jsonl(run_dir / "attack_steps.jsonl"), "attack_step_id"))

    retrieval_rows = read_jsonl(run_dir / "retrieval_results.jsonl")
    for idx, record in enumerate(retrieval_rows, start=1):
        results = record.get("results", [])
        if retrieval_enabled and not results:
            rows.append(_row(run_id, "WARNING", "retrieval", f"Retrieval record {idx} has no results", "retrieval_results.jsonl"))
        for result in results:
            chunk_id = str(result.get("chunk_id", ""))
            if chunk_id and chunk_id not in chunk_ids:
                rows.append(_row(run_id, "WARNING", "retrieval", f"Retrieved unknown chunk_id: {chunk_id}", "retrieval_results.jsonl"))

    if generation_enabled:
        generated_conditions = read_jsonl(run_dir / "generated_conditions.jsonl")
        verified_conditions = read_jsonl(run_dir / "verified_conditions.jsonl")
        final_conditions = read_jsonl(run_dir / "final_conditions.jsonl")
        rows.extend(_validate_condition_values(run_id, generated_conditions, "generated_conditions.jsonl", chunk_ids))
        rows.extend(_validate_condition_values(run_id, verified_conditions, "verified_conditions.jsonl", chunk_ids))
        rows.extend(_validate_condition_values(run_id, final_conditions, "final_conditions.jsonl", chunk_ids))

        if verification_enabled and len(verified_conditions) < len(generated_conditions):
            rows.append(
                _row(
                    run_id,
                    "WARNING",
                    "verification",
                    "Fewer verified conditions than generated conditions",
                    "verified_conditions.jsonl",
                )
            )

    rows.append(_row(run_id, "INFO", "summary", "Validation completed", str(run_dir)))
    return rows


def _validate_jsonl_file(
    run_id: str,
    run_dir: Path,
    filename: str,
    required_fields: list[str],
    *,
    required: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = run_dir / filename

    if not path.exists():
        level = "ERROR" if required else "WARNING"
        rows.append(_row(run_id, level, filename, f"Missing {filename}", str(path)))
        return rows

    records = read_jsonl(path)
    if required and not records:
        rows.append(_row(run_id, "WARNING", filename, f"{filename} is empty", str(path)))
        return rows

    for idx, record in enumerate(records[:50], start=1):
        for field in required_fields:
            if field not in record:
                rows.append(_row(run_id, "ERROR", filename, f"Record {idx} missing field: {field}", str(path)))

    rows.append(_row(run_id, "INFO", filename, f"Rows: {len(records)}", str(path)))
    return rows


def _validate_unique_ids(run_id: str, artifact: str, rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        value = str(row.get(field, ""))
        if not value:
            continue
        if value in seen:
            output.append(_row(run_id, "WARNING", artifact, f"Duplicate {field}: {value}", artifact))
        seen.add(value)
    return output


def _validate_condition_values(
    run_id: str,
    conditions: list[dict[str, Any]],
    filename: str,
    known_chunk_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        cid = condition.get("condition_id", "<missing>")
        _check_allowed(rows, run_id, filename, cid, "condition_type", condition.get("condition_type"), VALID_CONDITION_TYPES)
        _check_allowed(rows, run_id, filename, cid, "category", condition.get("category"), VALID_CATEGORIES)
        _check_allowed(rows, run_id, filename, cid, "support_level", condition.get("support_level"), VALID_SUPPORT)
        _check_allowed(rows, run_id, filename, cid, "directness", condition.get("directness"), VALID_DIRECTNESS)
        _check_allowed(rows, run_id, filename, cid, "status", condition.get("status"), VALID_STATUS)

        statement = str(condition.get("statement", "")).strip()
        if not statement:
            rows.append(_row(run_id, "ERROR", filename, f"Condition {cid} has empty statement", filename))
        if len(statement.split()) > 25:
            rows.append(_row(run_id, "WARNING", filename, f"Condition {cid} is longer than 25 words", filename))

        evidence_ids = condition.get("evidence_chunk_ids", [])
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        for chunk_id in evidence_ids:
            if chunk_id and str(chunk_id) not in known_chunk_ids:
                rows.append(_row(run_id, "WARNING", filename, f"Condition {cid} cites unknown chunk_id: {chunk_id}", filename))
    return rows


def _check_allowed(
    rows: list[dict[str, Any]],
    run_id: str,
    filename: str,
    condition_id: Any,
    field: str,
    value: Any,
    allowed: set[str],
) -> None:
    text = str(value).strip().lower()
    if text not in allowed:
        rows.append(_row(run_id, "ERROR", filename, f"Condition {condition_id} invalid {field}: {value!r}", filename))


def _discover_run_dirs(runs_root: Path, selected_runs: list[str] | None) -> list[Path]:
    if selected_runs:
        return [runs_root / run_id for run_id in selected_runs if (runs_root / run_id).exists()]
    return sorted(path for path in runs_root.iterdir() if path.is_dir())


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _row(run_id: str, level: str, artifact: str, message: str, path: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "level": level,
        "artifact": artifact,
        "message": message,
        "path": path,
    }


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Validate completed annotation CSV files.

The script checks required columns, allowed label values, missing annotator IDs,
and duplicate item/annotator pairs. It writes a CSV validation report.

Example:
    python scripts/validate_annotations.py --annotation-dir data/annotation/completed --out outputs/validation/annotation_validation.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.common import collect_csvs, read_csv_rows, write_csv_rows


CONDITION_LABELS = {
    "annotator_valid": {"yes", "no", "unclear"},
    "annotator_grounded": {"yes", "partially", "partial", "no"},
    "annotator_directness": {"direct", "indirect", "campaign_level"},
    "annotator_redundant": {"yes", "no"},
    "annotator_category_correct": {"yes", "no", "unclear"},
}
DESCRIPTION_LABELS = {
    "annotator_faithfulness": {"faithful", "partially_faithful", "partially", "unfaithful"},
    "annotator_specificity": {"specific", "generic"},
    "annotator_grounded": {"explicit", "inferred", "unsupported", "yes", "partially", "partial", "no"},
}
RETRIEVAL_LABELS = {
    "annotator_relevant": {"yes", "partially", "partial", "no"},
    "annotator_evidence_type": {"direct_step_evidence", "supporting_context", "campaign_context", "irrelevant"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate completed annotation CSV files.")
    parser.add_argument("--annotation-dir", action="append", default=[], help="Directory containing annotation CSVs.")
    parser.add_argument("--file", action="append", default=[], help="Explicit annotation CSV file. Can be repeated.")
    parser.add_argument("--out", default="outputs/validation/annotation_validation.csv", help="Validation report CSV path.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with code 1 if any ERROR is found.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = [Path(p) for p in args.file]
    files.extend(collect_csvs([Path(p) for p in args.annotation_dir], ["*.csv"]))
    files = _unique(files)

    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(validate_file(path))

    write_csv_rows(args.out, rows)
    num_errors = sum(1 for row in rows if row.get("level") == "ERROR")
    num_warnings = sum(1 for row in rows if row.get("level") == "WARNING")
    print(f"Validated files: {len(files)}")
    print(f"Errors: {num_errors}; Warnings: {num_warnings}")
    print(f"Report: {args.out}")

    if args.fail_on_error and num_errors:
        raise SystemExit(1)


def validate_file(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    report: list[dict[str, Any]] = []
    task = _infer_task(path, rows)

    if not rows:
        return [_row(path, "WARNING", task, "File is empty", "")]

    if task == "conditions":
        report.extend(_validate_task(path, rows, task, ["run_id", "condition_id"], CONDITION_LABELS))
    elif task == "descriptions":
        report.extend(_validate_task(path, rows, task, ["run_id", "attack_step_id"], DESCRIPTION_LABELS))
    elif task == "retrieval":
        report.extend(_validate_task(path, rows, task, ["run_id", "case_id", "attack_step_id", "chunk_id"], RETRIEVAL_LABELS))
    else:
        report.append(_row(path, "WARNING", task, "Could not infer annotation task from filename/columns", ""))

    report.append(_row(path, "INFO", task, f"Rows: {len(rows)}", ""))
    return report


def _validate_task(
    path: Path,
    rows: list[dict[str, str]],
    task: str,
    key_fields: list[str],
    label_allowed: dict[str, set[str]],
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    fieldnames = set(rows[0].keys())

    required = set(key_fields) | set(label_allowed.keys()) | {"annotator_id"}
    for field in sorted(required):
        if field not in fieldnames:
            report.append(_row(path, "ERROR", task, f"Missing required column: {field}", field))

    seen_item_annotator: set[tuple[str, str]] = set()
    for row_index, row in enumerate(rows, start=2):
        annotator_id = str(row.get("annotator_id", "")).strip()
        if not annotator_id:
            report.append(_row(path, "WARNING", task, f"Row {row_index} has empty annotator_id", "annotator_id"))

        item_key = tuple(str(row.get(field, "")).strip() for field in key_fields)
        if not all(item_key):
            report.append(_row(path, "ERROR", task, f"Row {row_index} has incomplete item key", ";".join(key_fields)))
        else:
            pair = ("|".join(item_key), annotator_id)
            if annotator_id and pair in seen_item_annotator:
                report.append(_row(path, "WARNING", task, f"Duplicate item/annotator pair at row {row_index}", pair[0]))
            seen_item_annotator.add(pair)

        for field, allowed in label_allowed.items():
            value = str(row.get(field, "")).strip().lower()
            if not value:
                report.append(_row(path, "WARNING", task, f"Row {row_index} has empty label: {field}", field))
            elif value not in allowed:
                report.append(_row(path, "ERROR", task, f"Row {row_index} invalid {field}: {value!r}", field))

    return report


def _infer_task(path: Path, rows: list[dict[str, str]]) -> str:
    name = path.name.lower()
    if "condition" in name:
        return "conditions"
    if "description" in name:
        return "descriptions"
    if "retrieval" in name:
        return "retrieval"

    if rows:
        columns = set(rows[0].keys())
        if "condition_id" in columns:
            return "conditions"
        if "description" in columns and "annotator_faithfulness" in columns:
            return "descriptions"
        if "chunk_id" in columns and "annotator_relevant" in columns:
            return "retrieval"
    return "unknown"


def _unique(paths: list[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen and path.exists():
            seen.add(resolved)
            output.append(path)
    return output


def _row(path: Path, level: str, task: str, message: str, field: str) -> dict[str, Any]:
    return {
        "file": str(path),
        "level": level,
        "task": task,
        "message": message,
        "field": field,
    }


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Aggregate per-case metrics.json files produced by app3.py.

The script is intentionally schema-tolerant: it does not require every case to
have exactly the same metric keys. It flattens all primitive JSON values into a
per-case CSV and builds an overall JSON summary with:

- summed count-like metrics;
- macro averages for rate/coverage/average metrics;
- simple numeric summaries for every numeric metric;
- a few micro-rates when the required numerators/denominators are available.

Expected input layout:

    output/<case-id>/metrics.json

Example:

    python scripts/aggregate_metrics.py \
      --input-root output \
      --output-dir evaluation_final_metrics

Outputs:

    evaluation_final_metrics/automatic_metrics_by_case.csv
    evaluation_final_metrics/automatic_metrics_numeric_summary.csv
    evaluation_final_metrics/automatic_metrics_overall.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MISSING = "NA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate all output/<case-id>/metrics.json files into CSV and JSON summaries."
    )
    parser.add_argument(
        "--input-root",
        default="output",
        help="Root directory containing one subdirectory per case. Default: output",
    )
    parser.add_argument(
        "--glob",
        default="*/metrics.json",
        help="Glob pattern relative to --input-root. Default: */metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_final_metrics",
        help="Directory where aggregate outputs are written. Default: evaluation_final_metrics",
    )
    parser.add_argument(
        "--missing",
        default=DEFAULT_MISSING,
        help="String used for missing values in CSV outputs. Default: NA",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation for JSON output. Default: 2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_files = sorted(input_root.glob(args.glob))
    if not metrics_files:
        raise FileNotFoundError(
            f"No metrics.json files found with pattern {input_root / args.glob}. "
            "Check --input-root and --glob."
        )

    case_records: list[dict[str, Any]] = []
    raw_metrics_by_case: dict[str, Any] = {}
    warnings: list[str] = []

    for metrics_path in metrics_files:
        metrics = read_json(metrics_path)
        case_id = infer_case_id(metrics, metrics_path)

        if case_id in raw_metrics_by_case:
            raise ValueError(
                f"Duplicate case_id '{case_id}' found. Previous and current file include the same case ID: {metrics_path}"
            )

        flat = flatten_json(metrics)
        flat["case_id"] = case_id
        flat["metrics_file"] = str(metrics_path)

        case_records.append(flat)
        raw_metrics_by_case[case_id] = metrics

    by_case_path = output_dir / "automatic_metrics_by_case.csv"
    write_flat_csv(by_case_path, case_records, missing=args.missing)

    numeric_summary = build_numeric_summary(case_records)
    numeric_summary_path = output_dir / "automatic_metrics_numeric_summary.csv"
    write_numeric_summary_csv(numeric_summary_path, numeric_summary)

    overall = build_overall_summary(
        case_records=case_records,
        raw_metrics_by_case=raw_metrics_by_case,
        metrics_files=metrics_files,
        numeric_summary=numeric_summary,
        warnings=warnings,
    )

    overall_path = output_dir / "automatic_metrics_overall.json"
    write_json(overall_path, overall, indent=args.indent)

    print(f"Aggregated {len(case_records)} case metrics files.")
    print(f"- {by_case_path}")
    print(f"- {numeric_summary_path}")
    print(f"- {overall_path}")


# ---------------------------------------------------------------------------
# JSON and flattening utilities
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write("\n")


def infer_case_id(metrics: Any, metrics_path: Path) -> str:
    """Find a stable case id from the JSON content, falling back to folder name."""
    if isinstance(metrics, dict):
        for key in ("case_id", "case", "id"):
            value = metrics.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        metadata = metrics.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get("case_id")
            if isinstance(value, str) and value.strip():
                return value.strip()

    # Expected layout: output/<case-id>/metrics.json
    return metrics_path.parent.name


def flatten_json(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries using dot-separated keys.

    Lists and dictionaries inside lists are serialized as JSON strings because
    they do not map cleanly to scalar CSV columns.
    """
    out: dict[str, Any] = {}

    if isinstance(value, dict):
        for key, child in value.items():
            clean_key = str(key).strip()
            path = f"{prefix}.{clean_key}" if prefix else clean_key
            out.update(flatten_json(child, path))
        return out

    if isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=False)
        return out

    out[prefix] = value
    return out


def write_flat_csv(path: Path, records: list[dict[str, Any]], *, missing: str) -> None:
    if not records:
        raise ValueError("No records to write.")

    # Keep identifiers first, then all metric columns in stable alphabetical order.
    first_cols = ["case_id", "metrics_file"]
    all_cols = sorted({key for row in records for key in row.keys()})
    columns = first_cols + [col for col in all_cols if col not in first_cols]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({col: csv_value(row.get(col, missing), missing=missing) for col in columns})


# ---------------------------------------------------------------------------
# Numeric aggregation
# ---------------------------------------------------------------------------


def build_numeric_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute generic numeric summaries for every numeric column.

    This is deliberately conservative: it does not pretend every value is a
    micro-average. It reports per-column count, sum, mean, median, min, and max.
    """
    summary: dict[str, dict[str, Any]] = {}

    keys = sorted({key for row in records for key in row.keys()})
    for key in keys:
        if key in {"case_id", "metrics_file"}:
            continue

        values = [to_number(row.get(key)) for row in records]
        nums = [value for value in values if value is not None and math.isfinite(value)]
        if not nums:
            continue

        summary[key] = {
            "n": len(nums),
            "missing": len(records) - len(nums),
            "sum": float(sum(nums)),
            "mean": float(statistics.fmean(nums)),
            "median": float(statistics.median(nums)),
            "min": float(min(nums)),
            "max": float(max(nums)),
        }

    return summary


def write_numeric_summary_csv(path: Path, summary: dict[str, dict[str, Any]]) -> None:
    columns = ["metric", "n", "missing", "sum", "mean", "median", "min", "max"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for metric, stats in sorted(summary.items()):
            row = {"metric": metric}
            row.update(stats)
            writer.writerow(row)


def build_overall_summary(
    *,
    case_records: list[dict[str, Any]],
    raw_metrics_by_case: dict[str, Any],
    metrics_files: list[Path],
    numeric_summary: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    count_sums = aggregate_count_like_metrics(numeric_summary)
    macro_averages = aggregate_macro_average_metrics(numeric_summary)
    weighted_averages = compute_known_weighted_averages(case_records, warnings)
    micro_rates = compute_known_micro_rates(case_records, warnings)

    return {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "num_cases": len(case_records),
            "case_ids": sorted(raw_metrics_by_case.keys()),
            "metrics_files": [str(path) for path in metrics_files],
            "aggregation_notes": [
                "Per-case flattened values are written to automatic_metrics_by_case.csv.",
                "Count-like metrics are summed when their key name suggests a count/total.",
                "Rates, coverage values, ratios, and averages are macro-averaged by default.",
                "Micro-rates are computed only when the required numerators and denominators are available.",
            ],
        },
        "count_sums": count_sums,
        "macro_averages": macro_averages,
        "weighted_averages": weighted_averages,
        "micro_rates": micro_rates,
        "numeric_summary": numeric_summary,
        "warnings": warnings,
    }


def aggregate_count_like_metrics(numeric_summary: dict[str, dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, stats in numeric_summary.items():
        if is_count_like_key(key):
            out[key] = stats["sum"]
    return out


def aggregate_macro_average_metrics(numeric_summary: dict[str, dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, stats in numeric_summary.items():
        if is_rate_or_average_key(key):
            out[key] = stats["mean"]
    return out


def compute_known_weighted_averages(records: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    """Compute selected weighted averages when known denominators exist.

    These mappings are optional. If the corresponding keys are not present in
    your metrics.json files, the function records a warning and skips them.
    """
    mappings = {
        # Average evidence count per generated field type.
        "automatic_metrics.average_evidence_count.descriptions": "counts.generated_descriptions",
        "automatic_metrics.average_evidence_count.preconditions": "counts.generated_preconditions",
        "automatic_metrics.average_evidence_count.postconditions": "counts.generated_postconditions",
        # Average field length per generated field type.
        "automatic_metrics.average_field_length_tokens.descriptions": "counts.generated_descriptions",
        "automatic_metrics.average_field_length_tokens.preconditions": "counts.generated_preconditions",
        "automatic_metrics.average_field_length_tokens.postconditions": "counts.generated_postconditions",
        # Runtime and context metrics, if available.
        "automatic_metrics.retrieval_transferability.average_retrieved_context_tokens": "counts.attack_steps",
        "automatic_metrics.retrieval_transferability.average_retrieved_chunks_per_step": "counts.attack_steps",
        "automatic_metrics.latency_ms.average_retrieval_per_attack_step": "counts.attack_steps",
        "automatic_metrics.latency_ms.average_generation_per_attack_step": "counts.attack_steps",
    }

    out: dict[str, Any] = {}
    for value_key, weight_key in mappings.items():
        if not any(value_key in row for row in records):
            continue

        weighted = weighted_mean(records, value_key=value_key, weight_key=weight_key)
        if weighted is None:
            warnings.append(
                f"Could not compute weighted average for {value_key}: missing or zero denominator {weight_key}."
            )
            continue

        out[value_key] = {
            "weighted_mean": weighted["mean"],
            "weight_key": weight_key,
            "total_weight": weighted["total_weight"],
            "n_cases_used": weighted["n_cases_used"],
        }

    return out


def compute_known_micro_rates(records: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    """Compute paper-useful micro-rates when possible.

    These are intentionally limited to rates whose numerator and denominator can
    be reconstructed from common app3.py metrics. Missing inputs are reported in
    warnings rather than silently guessed.
    """
    out: dict[str, Any] = {}

    # Description coverage usually equals generated descriptions / attack steps
    # because the pipeline generates at most one enriched description per step.
    add_micro_rate(
        out,
        records,
        name="field_coverage.descriptions",
        numerator_key="counts.generated_descriptions",
        denominator_key="counts.attack_steps",
        warnings=warnings,
    )

    # Duplicate-condition micro-rates, if removed counts are available.
    add_duplicate_micro_rate(
        out,
        records,
        name="duplicate_condition_rate.preconditions",
        generated_key="counts.generated_preconditions",
        removed_key="automatic_metrics.duplicate_condition_rate.removed_preconditions",
        warnings=warnings,
    )
    add_duplicate_micro_rate(
        out,
        records,
        name="duplicate_condition_rate.postconditions",
        generated_key="counts.generated_postconditions",
        removed_key="automatic_metrics.duplicate_condition_rate.removed_postconditions",
        warnings=warnings,
    )
    add_duplicate_micro_rate(
        out,
        records,
        name="duplicate_condition_rate.total",
        generated_key="counts.generated_conditions_total",
        removed_key="automatic_metrics.duplicate_condition_rate.removed_total",
        warnings=warnings,
    )

    return out


def add_micro_rate(
    out: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    name: str,
    numerator_key: str,
    denominator_key: str,
    warnings: list[str],
) -> None:
    numerator = sum_numbers(records, numerator_key)
    denominator = sum_numbers(records, denominator_key)

    if denominator is None or denominator == 0:
        warnings.append(
            f"Could not compute micro-rate {name}: missing or zero denominator {denominator_key}."
        )
        return

    if numerator is None:
        warnings.append(f"Could not compute micro-rate {name}: missing numerator {numerator_key}.")
        return

    out[name] = {
        "numerator_key": numerator_key,
        "denominator_key": denominator_key,
        "numerator": numerator,
        "denominator": denominator,
        "micro_rate": numerator / denominator,
    }


def add_duplicate_micro_rate(
    out: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    name: str,
    generated_key: str,
    removed_key: str,
    warnings: list[str],
) -> None:
    generated = sum_numbers(records, generated_key)
    removed = sum_numbers(records, removed_key)

    if removed is None:
        # Duplicate-removal metrics may not exist in older runs.
        return

    generated = generated or 0.0
    denominator = generated + removed
    if denominator == 0:
        warnings.append(f"Could not compute micro-rate {name}: generated + removed is zero.")
        return

    out[name] = {
        "generated_key": generated_key,
        "removed_key": removed_key,
        "generated_after_filtering": generated,
        "removed": removed,
        "denominator_generated_before_filtering": denominator,
        "micro_rate": removed / denominator,
    }


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def is_count_like_key(key: str) -> bool:
    lower = key.lower()
    parts = lower.replace(".", "_").split("_")

    if lower in {"case_id", "metrics_file"}:
        return False

    if any(token in parts for token in ["count", "counts", "num", "n"]):
        return True

    if lower.startswith("counts."):
        return True

    if lower.endswith("_total") or lower.endswith(".total"):
        return True

    if "removed_" in lower or lower.endswith("removed"):
        return True

    return False


def is_rate_or_average_key(key: str) -> bool:
    lower = key.lower()
    keywords = [
        "rate",
        "coverage",
        "ratio",
        "average",
        "avg",
        "mean",
        "median",
        "p95",
        "latency",
        "tokens",
    ]
    return any(keyword in lower for keyword in keywords) and not is_count_like_key(key)


# ---------------------------------------------------------------------------
# Numeric and CSV utilities
# ---------------------------------------------------------------------------


def to_number(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        # Avoid treating True/False as 1/0 for metric aggregation.
        return None

    if isinstance(value, (int, float)):
        if math.isnan(value) if isinstance(value, float) else False:
            return None
        return float(value)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.upper() == DEFAULT_MISSING:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None

    return None


def sum_numbers(records: list[dict[str, Any]], key: str) -> float | None:
    seen = False
    total = 0.0
    for row in records:
        value = to_number(row.get(key))
        if value is None:
            continue
        seen = True
        total += value
    return total if seen else None


def weighted_mean(records: list[dict[str, Any]], *, value_key: str, weight_key: str) -> dict[str, Any] | None:
    total_weighted_value = 0.0
    total_weight = 0.0
    n_cases_used = 0

    for row in records:
        value = to_number(row.get(value_key))
        weight = to_number(row.get(weight_key))
        if value is None or weight is None or weight <= 0:
            continue
        total_weighted_value += value * weight
        total_weight += weight
        n_cases_used += 1

    if total_weight == 0:
        return None

    return {
        "mean": total_weighted_value / total_weight,
        "total_weight": total_weight,
        "n_cases_used": n_cases_used,
    }


def csv_value(value: Any, *, missing: str) -> str:
    if value is None:
        return missing
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


if __name__ == "__main__":
    main()
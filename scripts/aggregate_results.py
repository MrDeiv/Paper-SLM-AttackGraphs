#!/usr/bin/env python3
"""
Aggregate automatic run outputs and optional annotation scores.

This script summarizes:
- generated/verified/final condition counts
- accepted condition yield
- verification rejection rate
- runtime metrics
- run configuration metadata

It can also copy or merge annotation score tables produced by
scripts/score_annotations.py into a final summary directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.common import read_csv_rows, write_csv_rows
from src.evaluation.runtime_metrics import compute_runtime_metrics
from src.io_utils import read_json, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate run outputs and metric tables.")
    parser.add_argument("--runs", default="outputs/runs", help="Directory containing run outputs.")
    parser.add_argument("--scores", default="outputs/final_metrics", help="Directory containing annotation score CSVs.")
    parser.add_argument("--out", default="outputs/final_metrics", help="Output directory for aggregate tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs)
    scores_dir = Path(args.scores)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir() and (path / "config.json").exists())
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with config.json found under {runs_root}")

    run_metadata = [_run_metadata(run_dir) for run_dir in run_dirs]
    condition_counts = [_condition_counts(run_dir) for run_dir in run_dirs]

    runtime_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        runtime_rows.extend(read_jsonl(run_dir / "runtime_metrics.jsonl"))
    runtime_metrics = compute_runtime_metrics(runtime_rows) if runtime_rows else []

    write_csv_rows(out_dir / "run_metadata.csv", run_metadata)
    write_csv_rows(out_dir / "automatic_condition_counts_by_variant.csv", _merge_by_run(run_metadata, condition_counts))
    write_csv_rows(out_dir / "runtime_by_variant.csv", _merge_by_run(run_metadata, runtime_metrics))

    final_summary = _build_final_summary(out_dir, scores_dir, run_metadata, condition_counts, runtime_metrics)
    write_csv_rows(out_dir / "final_metrics_summary.csv", final_summary)

    print(f"Aggregated runs: {len(run_dirs)}")
    print(f"Aggregate outputs: {out_dir}")


def _run_metadata(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    retrieval = config.get("retrieval", {})
    chunking = config.get("chunking", {})
    generation = config.get("generation", {})
    verification = config.get("verification", {})
    models = config.get("models", {})

    return {
        "run_id": config.get("run", {}).get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "retriever": retrieval.get("strategy", "unknown"),
        "retrieval_enabled": retrieval.get("enabled", None),
        "top_k": retrieval.get("top_k", None),
        "chunking_strategy": chunking.get("strategy", "unknown"),
        "chunk_size": chunking.get("max_chunk_tokens", None),
        "chunk_overlap": chunking.get("overlap_tokens", None),
        "generation_enabled": generation.get("enabled", None),
        "verification_enabled": verification.get("enabled", None),
        "text_generation_model": models.get("text_generation", ""),
        "embedding_model": models.get("embeddings", ""),
    }


def _condition_counts(run_dir: Path) -> dict[str, Any]:
    generated = read_jsonl(run_dir / "generated_conditions.jsonl")
    verified = read_jsonl(run_dir / "verified_conditions.jsonl")
    final = read_jsonl(run_dir / "final_conditions.jsonl")

    rejected = [row for row in verified if row.get("status") == "rejected"]
    accepted = [row for row in verified if row.get("status") == "accepted"]
    unsupported_rejected = [row for row in rejected if row.get("support_level") == "unsupported"]
    campaign_rejected = [row for row in rejected if row.get("directness") == "campaign_level"]

    generated_n = len(generated)
    verified_n = len(verified)
    final_n = len(final)

    return {
        "run_id": run_dir.name,
        "num_generated_conditions": generated_n,
        "num_verified_conditions": verified_n,
        "num_final_conditions": final_n,
        "num_accepted_verified_conditions": len(accepted),
        "num_rejected_verified_conditions": len(rejected),
        "num_rejected_unsupported": len(unsupported_rejected),
        "num_rejected_campaign_level": len(campaign_rejected),
        "accepted_condition_yield": _safe_divide(final_n, generated_n),
        "verification_rejection_rate": _safe_divide(len(rejected), verified_n),
    }


def _build_final_summary(
    out_dir: Path,
    scores_dir: Path,
    run_metadata: list[dict[str, Any]],
    condition_counts: list[dict[str, Any]],
    runtime_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = _merge_by_run(run_metadata, condition_counts, runtime_metrics)

    for filename in [
        "condition_quality_by_variant.csv",
        "description_quality_by_variant.csv",
        "retrieval_quality_by_variant.csv",
    ]:
        table_path = scores_dir / filename
        if table_path.exists():
            summary = _merge_by_run(summary, read_csv_rows(table_path))

    return summary


def _merge_by_run(*tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for table in tables:
        for row in table:
            run_id = str(row.get("run_id", ""))
            if not run_id:
                continue
            if run_id not in merged:
                merged[run_id] = {"run_id": run_id}
                order.append(run_id)
            merged[run_id].update(row)

    return [merged[run_id] for run_id in order]


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


if __name__ == "__main__":
    main()
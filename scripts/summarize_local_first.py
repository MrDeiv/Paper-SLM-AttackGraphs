#!/usr/bin/env python3
"""
Summarize local-first metrics across experiment runs.

The script reads outputs/runs/<run_id>/metrics/local_first_metrics.json files and
writes a flat CSV table suitable for the paper's reproducibility/local-first
resource table.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize local-first metrics across runs.")
    parser.add_argument("--runs", default="outputs/runs", help="Directory containing run output folders.")
    parser.add_argument("--out", default="outputs/final_metrics/local_first_by_variant.csv", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs)
    rows: list[dict[str, Any]] = []

    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics" / "local_first_metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(_flatten_local_first(metrics))

    write_csv_rows(args.out, rows)
    print(f"Local-first rows: {len(rows)}")
    print(f"Output: {args.out}")


def _flatten_local_first(metrics: dict[str, Any]) -> dict[str, Any]:
    resource = metrics.get("resource_summary", {}) or {}
    artifact = metrics.get("artifact_metrics", {}) or {}
    model_inventory = metrics.get("model_inventory", {}) or {}
    config_summary = metrics.get("config_summary", {}) or {}

    return {
        "run_id": metrics.get("run_id", ""),
        "run_completed": metrics.get("run_completed", ""),
        "run_error": metrics.get("run_error", ""),
        "offline_run_declared": metrics.get("offline_run_declared", ""),
        "external_inference_api_configured": metrics.get("external_inference_api_configured", ""),
        "inference_backend": metrics.get("inference_backend", ""),
        "retriever": config_summary.get("retriever", ""),
        "chunking_strategy": config_summary.get("chunking_strategy", ""),
        "generation_enabled": config_summary.get("generation_enabled", ""),
        "verification_enabled": config_summary.get("verification_enabled", ""),
        "text_generation_model": config_summary.get("text_generation_model", ""),
        "embedding_model": config_summary.get("embedding_model", ""),
        "num_resource_samples": resource.get("num_resource_samples", ""),
        "elapsed_ms_observed": resource.get("elapsed_ms_observed", ""),
        "cpu_percent_system_mean": resource.get("cpu_percent_system_mean", ""),
        "cpu_percent_system_max": resource.get("cpu_percent_system_max", ""),
        "cpu_percent_process_mean": resource.get("cpu_percent_process_mean", ""),
        "cpu_percent_process_max": resource.get("cpu_percent_process_max", ""),
        "ram_total_mb": resource.get("ram_total_mb", ""),
        "ram_used_mb_mean": resource.get("ram_used_mb_mean", ""),
        "ram_used_mb_max": resource.get("ram_used_mb_max", ""),
        "process_rss_mb_mean": resource.get("process_rss_mb_mean", ""),
        "process_rss_mb_max": resource.get("process_rss_mb_max", ""),
        "gpu_name": resource.get("gpu_name", ""),
        "gpu_utilization_percent_mean": resource.get("gpu_utilization_percent_mean", ""),
        "gpu_utilization_percent_max": resource.get("gpu_utilization_percent_max", ""),
        "gpu_memory_used_mb_mean": resource.get("gpu_memory_used_mb_mean", ""),
        "gpu_memory_used_mb_max": resource.get("gpu_memory_used_mb_max", ""),
        "gpu_memory_total_mb_max": resource.get("gpu_memory_total_mb_max", ""),
        "network_observation_scope": resource.get("network_observation_scope", ""),
        "network_bytes_sent_delta": resource.get("network_bytes_sent_delta", ""),
        "network_bytes_recv_delta": resource.get("network_bytes_recv_delta", ""),
        "run_output_size_mb": artifact.get("run_output_size_mb", ""),
        "input_source_documents_size_mb": artifact.get("input_source_documents_size_mb", ""),
        "num_chunks_rows": artifact.get("num_chunks_rows", ""),
        "num_attack_steps_rows": artifact.get("num_attack_steps_rows", ""),
        "num_generated_conditions_rows": artifact.get("num_generated_conditions_rows", ""),
        "num_verified_conditions_rows": artifact.get("num_verified_conditions_rows", ""),
        "num_final_conditions_rows": artifact.get("num_final_conditions_rows", ""),
        "accepted_condition_yield": artifact.get("accepted_condition_yield", ""),
        "verification_rejection_rate": artifact.get("verification_rejection_rate", ""),
        "ollama_available": model_inventory.get("ollama_available", ""),
        "ollama_model_names": ";".join(model.get("name", "") for model in model_inventory.get("ollama_models", [])),
    }


if __name__ == "__main__":
    main()
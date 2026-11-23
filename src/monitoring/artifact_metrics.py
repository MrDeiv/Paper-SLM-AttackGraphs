"""
Artifact and disk-footprint metrics for local-first runs.

These helpers summarize local storage requirements and output completeness. The
metrics are intentionally file-based so they work across all pipeline variants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.io_utils import read_jsonl


JSONL_OUTPUT_FILES = {
    "chunks": "chunks.jsonl",
    "attack_steps": "attack_steps.jsonl",
    "retrieval_results": "retrieval_results.jsonl",
    "generated_descriptions": "generated_descriptions.jsonl",
    "generated_conditions": "generated_conditions.jsonl",
    "verified_conditions": "verified_conditions.jsonl",
    "final_conditions": "final_conditions.jsonl",
    "runtime_metrics": "runtime_metrics.jsonl",
    "resource_metrics": "resource_metrics.jsonl",
}


def collect_run_artifact_metrics(run_dir: str | Path) -> dict[str, Any]:
    """Collect artifact size and output-completeness metrics for one run."""
    run_dir = Path(run_dir)
    metrics: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_output_size_mb": _dir_size_mb(run_dir),
        "run_file_count": _file_count(run_dir),
    }

    for logical_name, filename in JSONL_OUTPUT_FILES.items():
        path = run_dir / filename
        metrics[f"{logical_name}_exists"] = path.exists()
        metrics[f"{logical_name}_size_mb"] = _file_size_mb(path)
        metrics[f"num_{logical_name}_rows"] = _count_jsonl_rows(path)

    chunks = read_jsonl(run_dir / "chunks.jsonl")
    source_paths = sorted({str(row.get("source_path", "")) for row in chunks if row.get("source_path")})
    existing_sources = [Path(path) for path in source_paths if Path(path).exists()]
    metrics["num_unique_source_documents"] = len(source_paths)
    metrics["input_source_documents_size_mb"] = sum(_file_size_mb(path) for path in existing_sources)

    retrieval_records = read_jsonl(run_dir / "retrieval_results.jsonl")
    retrieved_chunk_ids = set()
    for record in retrieval_records:
        for result in record.get("results", []):
            chunk_id = str(result.get("chunk_id", ""))
            if chunk_id:
                retrieved_chunk_ids.add(chunk_id)
    metrics["num_unique_retrieved_chunks"] = len(retrieved_chunk_ids)

    generated_conditions = read_jsonl(run_dir / "generated_conditions.jsonl")
    verified_conditions = read_jsonl(run_dir / "verified_conditions.jsonl")
    final_conditions = read_jsonl(run_dir / "final_conditions.jsonl")
    metrics["accepted_condition_yield"] = _safe_divide(len(final_conditions), len(generated_conditions))
    metrics["verification_rejection_rate"] = _safe_divide(
        len([row for row in verified_conditions if row.get("status") == "rejected"]),
        len(verified_conditions),
    )

    return metrics


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total / (1024 * 1024)


def _file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for file_path in path.rglob("*") if file_path.is_file())


def _file_size_mb(path: Path) -> float:
    if not path.exists() or not path.is_file():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
"""
Retrieval metrics from flattened retrieval annotation CSV rows.

Each row should represent one retrieved chunk for one attack step. Expected
columns include:
- run_id
- case_id
- attack_step_id
- rank
- chunk_id
- annotator_relevant

The module computes precision@k and pooled recall@k. Pooled recall uses the
union of relevant chunks observed across all submitted systems for the same
(case_id, attack_step_id) pair.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.evaluation.common import group_by, is_filled, norm, safe_divide


K_VALUES = [3, 5, 10]


def compute_retrieval_metrics(rows: list[dict[str, Any]], k_values: list[int] | None = None) -> list[dict[str, Any]]:
    """Compute retrieval metrics grouped by run_id."""
    k_values = k_values or K_VALUES
    annotated_rows = [row for row in rows if is_filled(row.get("annotator_relevant"))]

    pooled_relevant = _build_pooled_relevance(annotated_rows)
    output: list[dict[str, Any]] = []

    for run_id, group in sorted(group_by(annotated_rows, lambda r: r.get("run_id", "unknown_run")).items()):
        output.append(_compute_for_run(str(run_id), group, pooled_relevant, k_values))

    return output


def _build_pooled_relevance(rows: list[dict[str, Any]]) -> dict[tuple[str, str], set[str]]:
    pooled: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        if _is_relevant(row, relaxed=False):
            key = (str(row.get("case_id", "")), str(row.get("attack_step_id", "")))
            chunk_id = str(row.get("chunk_id", ""))
            if chunk_id:
                pooled[key].add(chunk_id)
    return pooled


def _compute_for_run(
    run_id: str,
    rows: list[dict[str, Any]],
    pooled_relevant: dict[tuple[str, str], set[str]],
    k_values: list[int],
) -> dict[str, Any]:
    query_groups = group_by(rows, lambda r: (r.get("case_id", ""), r.get("attack_step_id", "")))
    result: dict[str, Any] = {
        "run_id": run_id,
        "num_annotated_retrieval_rows": len(rows),
        "num_annotated_queries": len(query_groups),
    }

    for k in k_values:
        precisions: list[float] = []
        relaxed_precisions: list[float] = []
        recalls: list[float] = []

        for query_key, query_rows in query_groups.items():
            top_rows = _top_k(query_rows, k)
            if not top_rows:
                continue

            strict_relevant = sum(1 for row in top_rows if _is_relevant(row, relaxed=False))
            relaxed_relevant = sum(1 for row in top_rows if _is_relevant(row, relaxed=True))
            precisions.append(strict_relevant / k)
            relaxed_precisions.append(relaxed_relevant / k)

            pool = pooled_relevant.get((str(query_key[0]), str(query_key[1])), set())
            if pool:
                retrieved_relevant_chunks = {
                    str(row.get("chunk_id", ""))
                    for row in top_rows
                    if str(row.get("chunk_id", "")) in pool
                }
                recalls.append(len(retrieved_relevant_chunks) / len(pool))

        result[f"evidence_precision_at_{k}"] = _mean(precisions)
        result[f"evidence_precision_at_{k}_relaxed"] = _mean(relaxed_precisions)
        result[f"pooled_evidence_recall_at_{k}"] = _mean(recalls)

    return result


def _top_k(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    def rank(row: dict[str, Any]) -> int:
        try:
            return int(row.get("rank", 999999))
        except Exception:
            return 999999

    return sorted(rows, key=rank)[:k]


def _is_relevant(row: dict[str, Any], relaxed: bool) -> bool:
    value = norm(row.get("annotator_relevant"))
    if value == "yes":
        return True
    if relaxed and value in {"partially", "partial"}:
        return True
    return False


def _mean(values: list[float]) -> float | None:
    return safe_divide(sum(values), len(values))
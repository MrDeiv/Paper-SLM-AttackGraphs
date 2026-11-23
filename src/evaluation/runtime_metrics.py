"""
Runtime aggregation helpers for structured CTI enrichment runs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.evaluation.common import group_by, safe_divide


def compute_runtime_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate runtime_metrics.jsonl rows by run_id."""
    output: list[dict[str, Any]] = []
    for run_id, group in sorted(group_by(rows, lambda r: r.get("run_id", "unknown_run")).items()):
        output.append(_compute_for_run(str(run_id), group))
    return output


def _compute_for_run(run_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed_by_stage: dict[str, list[float]] = defaultdict(list)
    elapsed_by_case: dict[str, float] = defaultdict(float)
    elapsed_by_step: dict[tuple[str, str], float] = defaultdict(float)

    for row in rows:
        stage = str(row.get("stage", "unknown"))
        elapsed = _float(row.get("elapsed_ms"))
        case_id = str(row.get("case_id", ""))
        attack_step_id = str(row.get("attack_step_id", ""))
        elapsed_by_stage[stage].append(elapsed)
        if case_id:
            elapsed_by_case[case_id] += elapsed
        if case_id and attack_step_id:
            elapsed_by_step[(case_id, attack_step_id)] += elapsed

    result: dict[str, Any] = {
        "run_id": run_id,
        "num_runtime_rows": len(rows),
        "num_cases_with_runtime": len(elapsed_by_case),
        "num_attack_steps_with_runtime": len(elapsed_by_step),
        "mean_runtime_per_case_ms": _mean(list(elapsed_by_case.values())),
        "median_runtime_per_case_ms": _median(list(elapsed_by_case.values())),
        "mean_runtime_per_attack_step_ms": _mean(list(elapsed_by_step.values())),
        "median_runtime_per_attack_step_ms": _median(list(elapsed_by_step.values())),
        "total_runtime_logged_ms": sum(_float(row.get("elapsed_ms")) for row in rows),
    }

    for stage, values in elapsed_by_stage.items():
        result[f"mean_{stage}_ms"] = _mean(values)
        result[f"median_{stage}_ms"] = _median(values)
        result[f"total_{stage}_ms"] = sum(values)

    return result


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _mean(values: list[float]) -> float | None:
    return safe_divide(sum(values), len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0
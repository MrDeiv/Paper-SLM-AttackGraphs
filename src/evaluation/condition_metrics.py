"""
Condition-level quality metrics from completed annotation CSV rows.

Expected annotation columns include:
- run_id
- annotator_valid
- annotator_grounded
- annotator_directness
- annotator_redundant
- annotator_category_correct
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.evaluation.common import group_by, is_filled, norm, safe_divide


def compute_condition_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute condition-quality metrics grouped by run_id."""
    output: list[dict[str, Any]] = []
    for run_id, group in sorted(group_by(rows, lambda r: r.get("run_id", "unknown_run")).items()):
        output.append(_compute_for_group(str(run_id), group))
    return output


def _compute_for_group(run_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_labels = [norm(row.get("annotator_valid")) for row in rows if is_filled(row.get("annotator_valid"))]
    grounded_labels = [norm(row.get("annotator_grounded")) for row in rows if is_filled(row.get("annotator_grounded"))]
    directness_labels = [norm(row.get("annotator_directness")) for row in rows if is_filled(row.get("annotator_directness"))]
    redundant_labels = [norm(row.get("annotator_redundant")) for row in rows if is_filled(row.get("annotator_redundant"))]
    category_labels = [norm(row.get("annotator_category_correct")) for row in rows if is_filled(row.get("annotator_category_correct"))]

    valid_counts = Counter(valid_labels)
    grounded_counts = Counter(grounded_labels)
    directness_counts = Counter(directness_labels)
    redundant_counts = Counter(redundant_labels)
    category_counts = Counter(category_labels)

    grounded_yes = grounded_counts.get("yes", 0)
    grounded_no = grounded_counts.get("no", 0)

    return {
        "run_id": run_id,
        "num_condition_rows": len(rows),
        "num_validity_annotated": len(valid_labels),
        "num_grounding_annotated": len(grounded_labels),
        "num_directness_annotated": len(directness_labels),
        "num_redundancy_annotated": len(redundant_labels),
        "num_category_annotated": len(category_labels),
        "condition_validity_precision": safe_divide(valid_counts.get("yes", 0), len(valid_labels)),
        "condition_grounding_rate_strict": safe_divide(grounded_yes, len(grounded_labels)),
        "condition_grounding_rate_relaxed": safe_divide(
            grounded_counts.get("yes", 0) + grounded_counts.get("partially", 0) + grounded_counts.get("partial", 0),
            len(grounded_labels),
        ),
        "unsupported_condition_rate": safe_divide(grounded_no, len(grounded_labels)),
        "redundancy_rate": safe_divide(redundant_counts.get("yes", 0), len(redundant_labels)),
        "direct_condition_rate": safe_divide(directness_counts.get("direct", 0), len(directness_labels)),
        "indirect_condition_rate": safe_divide(directness_counts.get("indirect", 0), len(directness_labels)),
        "campaign_level_condition_rate": safe_divide(directness_counts.get("campaign_level", 0), len(directness_labels)),
        "category_accuracy": safe_divide(category_counts.get("yes", 0), len(category_labels)),
        "valid_yes": valid_counts.get("yes", 0),
        "valid_no": valid_counts.get("no", 0),
        "valid_unclear": valid_counts.get("unclear", 0),
        "grounded_yes": grounded_counts.get("yes", 0),
        "grounded_partially": grounded_counts.get("partially", 0) + grounded_counts.get("partial", 0),
        "grounded_no": grounded_counts.get("no", 0),
        "directness_direct": directness_counts.get("direct", 0),
        "directness_indirect": directness_counts.get("indirect", 0),
        "directness_campaign_level": directness_counts.get("campaign_level", 0),
    }
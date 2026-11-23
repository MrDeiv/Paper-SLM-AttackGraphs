"""
Description-level quality metrics from completed annotation CSV rows.

Expected annotation columns include:
- run_id
- annotator_faithfulness
- annotator_specificity
- annotator_grounded
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.evaluation.common import group_by, is_filled, norm, safe_divide


def compute_description_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute description-quality metrics grouped by run_id."""
    output: list[dict[str, Any]] = []
    for run_id, group in sorted(group_by(rows, lambda r: r.get("run_id", "unknown_run")).items()):
        output.append(_compute_for_group(str(run_id), group))
    return output


def _compute_for_group(run_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    faithfulness_labels = [norm(row.get("annotator_faithfulness")) for row in rows if is_filled(row.get("annotator_faithfulness"))]
    specificity_labels = [norm(row.get("annotator_specificity")) for row in rows if is_filled(row.get("annotator_specificity"))]
    grounding_labels = [norm(row.get("annotator_grounded")) for row in rows if is_filled(row.get("annotator_grounded"))]

    faithfulness_counts = Counter(faithfulness_labels)
    specificity_counts = Counter(specificity_labels)
    grounding_counts = Counter(grounding_labels)

    faithful_count = faithfulness_counts.get("faithful", 0)
    partially_faithful_count = faithfulness_counts.get("partially_faithful", 0) + faithfulness_counts.get("partially", 0)

    explicit_count = grounding_counts.get("explicit", 0) + grounding_counts.get("yes", 0)
    inferred_count = grounding_counts.get("inferred", 0) + grounding_counts.get("partially", 0) + grounding_counts.get("partial", 0)
    unsupported_count = grounding_counts.get("unsupported", 0) + grounding_counts.get("no", 0)

    return {
        "run_id": run_id,
        "num_description_rows": len(rows),
        "num_faithfulness_annotated": len(faithfulness_labels),
        "num_specificity_annotated": len(specificity_labels),
        "num_grounding_annotated": len(grounding_labels),
        "description_faithfulness_rate_strict": safe_divide(faithful_count, len(faithfulness_labels)),
        "description_faithfulness_rate_relaxed": safe_divide(faithful_count + partially_faithful_count, len(faithfulness_labels)),
        "generic_description_rate": safe_divide(specificity_counts.get("generic", 0), len(specificity_labels)),
        "specific_description_rate": safe_divide(specificity_counts.get("specific", 0), len(specificity_labels)),
        "description_grounding_rate_strict": safe_divide(explicit_count, len(grounding_labels)),
        "description_grounding_rate_relaxed": safe_divide(explicit_count + inferred_count, len(grounding_labels)),
        "description_unsupported_rate": safe_divide(unsupported_count, len(grounding_labels)),
        "faithful": faithful_count,
        "partially_faithful": partially_faithful_count,
        "unfaithful": faithfulness_counts.get("unfaithful", 0),
        "specific": specificity_counts.get("specific", 0),
        "generic": specificity_counts.get("generic", 0),
        "grounding_explicit": explicit_count,
        "grounding_inferred": inferred_count,
        "grounding_unsupported": unsupported_count,
    }
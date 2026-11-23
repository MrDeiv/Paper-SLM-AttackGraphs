"""
Inter-annotator agreement metrics.

This module computes percent agreement and Cohen's kappa without requiring
scikit-learn. It expects annotation rows to contain an annotator_id column and a
stable item key such as condition_id, attack_step_id, or chunk_id.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.evaluation.common import is_filled, norm, safe_divide


def compute_agreement(
    rows: list[dict[str, Any]],
    *,
    task_name: str,
    item_key_fields: list[str],
    label_fields: list[str],
    annotator_field: str = "annotator_id",
) -> list[dict[str, Any]]:
    """
    Compute agreement for each label field on double-annotated items.
    """
    output: list[dict[str, Any]] = []
    for label_field in label_fields:
        pairs = _paired_labels(rows, item_key_fields, label_field, annotator_field)
        if not pairs:
            output.append(
                {
                    "annotation_task": task_name,
                    "label_name": label_field,
                    "num_double_annotated_items": 0,
                    "percent_agreement": None,
                    "cohens_kappa": None,
                }
            )
            continue

        matches = sum(1 for a, b in pairs if a == b)
        percent = safe_divide(matches, len(pairs))
        kappa = cohens_kappa(pairs)
        output.append(
            {
                "annotation_task": task_name,
                "label_name": label_field,
                "num_double_annotated_items": len(pairs),
                "percent_agreement": percent,
                "cohens_kappa": kappa,
            }
        )
    return output


def cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Compute Cohen's kappa for two categorical label sequences."""
    n = len(pairs)
    if n == 0:
        return None

    observed = sum(1 for a, b in pairs if a == b) / n
    labels_a = Counter(a for a, _ in pairs)
    labels_b = Counter(b for _, b in pairs)
    labels = set(labels_a) | set(labels_b)
    expected = sum((labels_a[label] / n) * (labels_b[label] / n) for label in labels)

    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def _paired_labels(
    rows: list[dict[str, Any]],
    item_key_fields: list[str],
    label_field: str,
    annotator_field: str,
) -> list[tuple[str, str]]:
    grouped: dict[tuple[str, ...], dict[str, str]] = defaultdict(dict)

    for row in rows:
        label = norm(row.get(label_field))
        annotator = str(row.get(annotator_field, "")).strip()
        if not annotator or not is_filled(label):
            continue
        item_key = tuple(str(row.get(field, "")) for field in item_key_fields)
        if not any(item_key):
            continue
        grouped[item_key][annotator] = label

    pairs: list[tuple[str, str]] = []
    for labels_by_annotator in grouped.values():
        if len(labels_by_annotator) < 2:
            continue
        annotators = sorted(labels_by_annotator)
        pairs.append((labels_by_annotator[annotators[0]], labels_by_annotator[annotators[1]]))
    return pairs
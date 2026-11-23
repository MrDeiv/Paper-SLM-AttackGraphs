#!/usr/bin/env python3
"""Compute inter-annotator agreement and final manual metrics.

Agreement is computed from the independent filled annotator files.  Final manual
metrics are computed only from adjudicated labels in adjudication_final.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final_eval_common import (
    allowed_labels,
    annotation_cfg,
    cohen_kappa,
    condition_criteria,
    description_criteria,
    get_eval_cfg,
    is_missing,
    load_config,
    mean,
    na_value,
    raw_agreement,
    read_csv,
    repo_path,
    safe_divide,
    split_values,
    validate_labels,
    validate_same_annotation_ids,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute agreement and adjudicated manual metrics.")
    parser.add_argument("--config", default="configs/evaluation/final_pipeline_eval.yaml")
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing labels for incomplete dry runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    eval_cfg = get_eval_cfg(config)
    ann_cfg = annotation_cfg(config)

    filled_dir = repo_path(eval_cfg.get("filled_annotation_dir", "annotation_round_1_filled"))
    adjudication_dir = repo_path(eval_cfg.get("adjudication_output_dir", "adjudication_round"))
    output_dir = repo_path(eval_cfg.get("final_metrics_output_dir", "evaluation_final_metrics"))
    output_dir.mkdir(parents=True, exist_ok=True)

    annotators = [str(x) for x in ann_cfg.get("annotators", ["A", "B"])]
    a_path = filled_dir / f"annotator_{annotators[0]}_filled.csv"
    b_path = filled_dir / f"annotator_{annotators[1]}_filled.csv"
    adjudicated_path = adjudication_dir / "adjudication_final.csv"

    a_rows = read_csv(a_path)
    b_rows = read_csv(b_path)
    adjudicated_rows = read_csv(adjudicated_path)

    if not a_rows or not b_rows:
        raise FileNotFoundError("Filled annotator files are required to compute agreement.")
    if not adjudicated_rows:
        raise FileNotFoundError(f"Adjudicated file is required: {adjudicated_path}")

    validate_same_annotation_ids(a_rows, b_rows)
    validate_labels(a_rows, config, allow_missing=args.allow_missing)
    validate_labels(b_rows, config, allow_missing=args.allow_missing)
    validate_adjudicated(adjudicated_rows, config, allow_missing=args.allow_missing)

    agreement = compute_agreement(a_rows, b_rows, config, str(a_path), str(b_path))
    condition_metrics = compute_condition_metrics(adjudicated_rows, config)
    description_metrics = compute_description_metrics(adjudicated_rows, config)
    failure_indicators = compute_failure_indicators(adjudicated_rows, config)
    transfer_metrics = compute_retrieval_transferability(adjudicated_rows, config)

    write_json(output_dir / "inter_annotator_agreement.json", agreement)
    write_json(output_dir / "manual_condition_metrics.json", condition_metrics)
    write_json(output_dir / "manual_description_metrics.json", description_metrics)
    write_json(output_dir / "failure_indicators.json", failure_indicators)
    write_json(output_dir / "retrieval_transferability_metrics.json", transfer_metrics)

    summary_rows = flatten_metric_tree("condition", condition_metrics) + flatten_metric_tree("description", description_metrics)
    summary_rows += flatten_metric_tree("failure", failure_indicators) + flatten_metric_tree("retrieval_transferability", transfer_metrics)
    write_csv(output_dir / "manual_metrics_summary.csv", summary_rows)

    print(f"Manual metrics written to: {output_dir}")


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


def compute_agreement(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], config: dict[str, Any], a_file: str, b_file: str) -> dict[str, Any]:
    a_map = {str(row["annotation_unit_id"]): row for row in a_rows}
    b_map = {str(row["annotation_unit_id"]): row for row in b_rows}
    ids = sorted(a_map)
    criteria_out: dict[str, Any] = {}
    weight_type = str(config.get("metrics", {}).get("cohen_kappa_weights", "quadratic"))

    for criterion in condition_criteria(config) + description_criteria(config):
        unit = "condition" if criterion in condition_criteria(config) else "description"
        scale = "binary" if criterion in annotation_cfg(config).get("binary_criteria", []) else "ordinal"
        y_a: list[str] = []
        y_b: list[str] = []
        for unit_id in ids:
            row_a = a_map[unit_id]
            row_b = b_map[unit_id]
            if str(row_a.get("field_type")) != unit:
                continue
            va = str(row_a.get(criterion, "")).strip()
            vb = str(row_b.get(criterion, "")).strip()
            if is_missing(va, config) or is_missing(vb, config):
                continue
            y_a.append(va)
            y_b.append(vb)

        weights = weight_type if scale == "ordinal" else None
        criteria_out[criterion] = {
            "unit": unit,
            "scale": scale,
            "statistic": "weighted_cohen_kappa" if scale == "ordinal" else "cohen_kappa",
            "kappa": cohen_kappa(y_a, y_b, weights=weights),
            "raw_agreement": raw_agreement(y_a, y_b),
            "n": len(y_a),
        }

    return {
        "metadata": {
            "annotator_a_file": a_file,
            "annotator_b_file": b_file,
            "n_condition_rows": sum(1 for r in a_rows if str(r.get("field_type")) == "condition"),
            "n_description_rows": sum(1 for r in a_rows if str(r.get("field_type")) == "description"),
            "ordinal_weighting": weight_type,
        },
        "criteria": criteria_out,
    }


# ---------------------------------------------------------------------------
# Final metrics from adjudicated labels
# ---------------------------------------------------------------------------


def compute_condition_metrics(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    condition_rows = [row for row in rows if str(row.get("field_type")) == "condition"]
    return {
        "preconditions": condition_group_metrics([r for r in condition_rows if str(r.get("condition_type")) == "precondition"], config),
        "postconditions": condition_group_metrics([r for r in condition_rows if str(r.get("condition_type")) == "postcondition"], config),
        "all_conditions": condition_group_metrics(condition_rows, config),
    }


def condition_group_metrics(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "condition_validity_strict": rate(rows, "condition_validity", {"2"}),
        "condition_validity_relaxed": rate(rows, "condition_validity", {"1", "2"}),
        "evidence_support_strict": rate(rows, "evidence_support", {"2"}),
        "evidence_support_relaxed": rate(rows, "evidence_support", {"1", "2"}),
        "direct_condition_strict": rate(rows, "directness", {"2"}),
        "direct_condition_relaxed": rate(rows, "directness", {"1", "2"}),
        "meaningfulness_rate": rate(rows, "meaningfulness", {"1"}),
        "atomicity_rate": rate(rows, "atomicity", {"1"}),
        "type_validity_strict": rate(rows, "type_validity", {"2"}),
        "type_validity_relaxed": rate(rows, "type_validity", {"1", "2"}),
        "non_redundancy_strict": rate(rows, "non_redundancy", {"2"}),
        "non_redundancy_relaxed": rate(rows, "non_redundancy", {"1", "2"}),
        "unsupported_condition_rate": rate(rows, "evidence_support", {"0"}),
        "generic_condition_rate": rate(rows, "meaningfulness", {"0"}),
        "campaign_level_condition_rate": rate(rows, "directness", {"0"}),
        "redundant_condition_rate": rate(rows, "non_redundancy", {"0"}),
    }


def compute_description_metrics(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    desc_rows = [row for row in rows if str(row.get("field_type")) == "description"]
    return {
        "all_descriptions": {
            "n": len(desc_rows),
            "description_faithfulness_strict": rate(desc_rows, "description_faithfulness", {"2"}),
            "description_faithfulness_relaxed": rate(desc_rows, "description_faithfulness", {"1", "2"}),
            "description_support_strict": rate(desc_rows, "description_evidence_support", {"2"}),
            "description_support_relaxed": rate(desc_rows, "description_evidence_support", {"1", "2"}),
            "description_specificity_strict": rate(desc_rows, "description_specificity", {"2"}),
            "description_specificity_relaxed": rate(desc_rows, "description_specificity", {"1", "2"}),
            "no_unsupported_claims_strict": rate(desc_rows, "unsupported_claims", {"2"}),
            "no_unsupported_claims_relaxed": rate(desc_rows, "unsupported_claims", {"1", "2"}),
            "unsupported_description_rate": rate(desc_rows, "unsupported_claims", {"0", "1"}),
        }
    }


def compute_failure_indicators(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    condition_rows = [row for row in rows if str(row.get("field_type")) == "condition"]
    desc_rows = [row for row in rows if str(row.get("field_type")) == "description"]
    return {
        "conditions": {
            "unsupported_condition_rate": rate(condition_rows, "evidence_support", {"0"}),
            "generic_condition_rate": rate(condition_rows, "meaningfulness", {"0"}),
            "campaign_level_condition_rate": rate(condition_rows, "directness", {"0"}),
            "redundant_condition_rate": rate(condition_rows, "non_redundancy", {"0"}),
            "wrong_type_rate": rate(condition_rows, "type_validity", {"0"}),
            "invalid_condition_rate": rate(condition_rows, "condition_validity", {"0"}),
        },
        "descriptions": {
            "unsupported_description_rate": rate(desc_rows, "unsupported_claims", {"0", "1"}),
            "generic_description_rate": rate(desc_rows, "description_specificity", {"0"}),
            "unfaithful_description_rate": rate(desc_rows, "description_faithfulness", {"0"}),
        },
    }


def compute_retrieval_transferability(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    condition_rows = [row for row in rows if str(row.get("field_type")) == "condition"]
    desc_rows = [row for row in rows if str(row.get("field_type")) == "description"]
    all_rows = condition_rows + desc_rows

    supported_rows = []
    first_support_ranks = []
    for row in all_rows:
        support_label = row.get("evidence_support_final") if row.get("field_type") == "condition" else row.get("description_evidence_support_final")
        if str(support_label) in {"1", "2"}:
            supported_rows.append(row)
            ranks = [int(x) for x in split_values(row.get("evidence_ranks", ""), config) if str(x).isdigit()]
            if ranks:
                first_support_ranks.append(min(ranks))

    steps = {str(row.get("attack_step_id")) for row in all_rows if row.get("attack_step_id")}
    steps_with_support = {str(row.get("attack_step_id")) for row in supported_rows if row.get("attack_step_id")}

    context_tokens = []
    for row in all_rows:
        try:
            context_tokens.append(float(row.get("context_tokens")))
        except Exception:
            pass

    unsupported_enrichments = 0
    for row in all_rows:
        if row.get("field_type") == "condition" and str(row.get("evidence_support_final")) == "0":
            unsupported_enrichments += 1
        if row.get("field_type") == "description" and str(row.get("description_evidence_support_final")) == "0":
            unsupported_enrichments += 1

    return {
        "evidence_coverage": safe_divide(len(steps_with_support), len(steps)),
        "mean_first_support_rank": mean(first_support_ranks),
        "condition_grounding_rate": rate(condition_rows, "evidence_support", {"1", "2"}),
        "unsupported_enrichment_rate": safe_divide(unsupported_enrichments, len(all_rows)),
        "average_retrieved_context_tokens": mean(context_tokens),
        "notes": [] if first_support_ranks else ["mean_first_support_rank is null because no valid evidence_ranks were available for supported rows."],
    }


def rate(rows: list[dict[str, Any]], column: str, positives: set[str]) -> float | None:
    """Rate over adjudicated labels.

    Adjudication files store final labels in <criterion>_final columns. The
    fallback to the base column keeps the helper usable with already-normalized
    rows.
    """
    if not rows:
        return None
    final_column = f"{column}_final"
    return sum(1 for row in rows if str(row.get(final_column, row.get(column, ""))).strip() in positives) / len(rows)


def validate_adjudicated(rows: list[dict[str, Any]], config: dict[str, Any], *, allow_missing: bool) -> None:
    for row in rows:
        field_type = str(row.get("field_type", ""))
        criteria = condition_criteria(config) if field_type == "condition" else description_criteria(config)
        unit_id = str(row.get("annotation_unit_id", ""))
        for criterion in criteria:
            final_col = f"{criterion}_final"
            value = str(row.get(final_col, "")).strip()
            if is_missing(value, config):
                if allow_missing:
                    continue
                raise ValueError(f"Missing adjudicated label {final_col} for {unit_id}")
            if value not in allowed_labels(config, criterion):
                raise ValueError(f"Invalid adjudicated label {final_col}={value!r} for {unit_id}")


def flatten_metric_tree(section: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        else:
            rows.append({"section": section, "metric": prefix, "value": value})

    visit("", data)
    return rows


if __name__ == "__main__":
    main()

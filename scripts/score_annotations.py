#!/usr/bin/env python3
"""
Score completed annotation CSV files.

This script reads completed annotation CSVs and writes aggregate metric tables:

- condition_quality_by_variant.csv
- description_quality_by_variant.csv
- retrieval_quality_by_variant.csv
- inter_annotator_agreement.csv

It accepts either explicit CSV files or directories containing completed CSVs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.agreement import compute_agreement
from src.evaluation.common import collect_csvs, rows_from_csvs, write_csv_rows
from src.evaluation.condition_metrics import compute_condition_metrics
from src.evaluation.description_metrics import compute_description_metrics
from src.evaluation.retrieval_metrics import compute_retrieval_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score completed annotation CSV files.")
    parser.add_argument(
        "--annotation-dir",
        action="append",
        default=[],
        help="Directory containing completed annotation CSVs. Can be repeated.",
    )
    parser.add_argument("--conditions", action="append", default=[], help="Completed condition annotation CSV.")
    parser.add_argument("--descriptions", action="append", default=[], help="Completed description annotation CSV.")
    parser.add_argument("--retrieval", action="append", default=[], help="Completed retrieval annotation CSV.")
    parser.add_argument("--out", default="outputs/final_metrics", help="Output directory for metric tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    search_roots = [Path(p) for p in args.annotation_dir]

    condition_files = _unique_paths(
        [Path(p) for p in args.conditions]
        + _exclude_templates(collect_csvs(search_roots, ["*conditions*.csv", "*condition*.csv"]))
    )

    description_files = _unique_paths(
        [Path(p) for p in args.descriptions]
        + _exclude_templates(collect_csvs(search_roots, ["*descriptions*.csv", "*description*.csv"]))
    )

    retrieval_files = _unique_paths(
        [Path(p) for p in args.retrieval]
        + _exclude_templates(collect_csvs(search_roots, ["*retrieval*.csv"]))
    )

    condition_rows = rows_from_csvs(condition_files)
    description_rows = rows_from_csvs(description_files)
    retrieval_rows = rows_from_csvs(retrieval_files)

    condition_metrics = compute_condition_metrics(condition_rows) if condition_rows else []
    description_metrics = compute_description_metrics(description_rows) if description_rows else []
    retrieval_metrics = compute_retrieval_metrics(retrieval_rows) if retrieval_rows else []

    write_csv_rows(out_dir / "condition_quality_by_variant.csv", condition_metrics)
    write_csv_rows(out_dir / "description_quality_by_variant.csv", description_metrics)
    write_csv_rows(out_dir / "retrieval_quality_by_variant.csv", retrieval_metrics)

    agreement_rows = []
    if condition_rows:
        agreement_rows.extend(
            compute_agreement(
                condition_rows,
                task_name="conditions",
                item_key_fields=["run_id", "condition_id"],
                label_fields=[
                    "annotator_valid",
                    "annotator_grounded",
                    "annotator_directness",
                    "annotator_redundant",
                    "annotator_category_correct",
                ],
            )
        )
    if description_rows:
        agreement_rows.extend(
            compute_agreement(
                description_rows,
                task_name="descriptions",
                item_key_fields=["run_id", "attack_step_id"],
                label_fields=["annotator_faithfulness", "annotator_specificity", "annotator_grounded"],
            )
        )
    if retrieval_rows:
        agreement_rows.extend(
            compute_agreement(
                retrieval_rows,
                task_name="retrieval",
                item_key_fields=["run_id", "case_id", "attack_step_id", "chunk_id"],
                label_fields=["annotator_relevant", "annotator_evidence_type"],
            )
        )
    write_csv_rows(out_dir / "inter_annotator_agreement.csv", agreement_rows)

    print(f"Condition files: {len(condition_files)}; rows: {len(condition_rows)}")
    print(f"Description files: {len(description_files)}; rows: {len(description_rows)}")
    print(f"Retrieval files: {len(retrieval_files)}; rows: {len(retrieval_rows)}")
    print(f"Metric outputs: {out_dir}")


def _exclude_templates(files: list[Path]) -> list[Path]:
    """Avoid scoring blank discovered template files."""
    excluded_tokens = {"to_annotate", "template"}
    return [path for path in files if not any(token in path.name.lower() for token in excluded_tokens)]


def _unique_paths(files: list[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        output.append(path)
    return output


if __name__ == "__main__":
    main()
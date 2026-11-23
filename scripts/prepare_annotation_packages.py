#!/usr/bin/env python3
"""Prepare grouped annotator packages for the CTI enrichment evaluation.

This is a second version of the annotation-packaging script.  It keeps the
annotation package readable by grouping rows by attack step:

    enriched description
    related preconditions
    related postconditions

The attack-step groups are still randomized with a deterministic seed, so the
annotators receive a randomized package while keeping the context needed to
understand each generated field.

Input
-----
Per-case annotation units, typically:

    evaluation_outputs/<case_id>/annotation_units.csv

Output
------

    annotation_round_1_grouped/
      annotation_master_unfilled.csv
      annotator_A.csv
      annotator_B.csv
      annotation_codebook.md
      row_manifest.json
      validation_report.json

Example
-------

    python scripts/prepare_annotation_grouped.py \
      --input-glob "evaluation_outputs/*/annotation_units.csv" \
      --output-dir annotation_round_1_grouped \
      --seed 42
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


MISSING_VALUE = "NA"

REQUIRED_COLUMNS = [
    "annotation_unit_id",
    "run_id",
    "case_id",
    "case_name",
    "milestone_id",
    "milestone_label",
    "tactic_id",
    "tactic_name",
    "attack_step_id",
    "attack_step_label",
    "technique_id",
    "technique_name",
    "field_type",
    "field_id",
    "condition_type",
    "original_description",
    "enriched_description",
    "generated_field_text",
    "evidence_chunk_ids",
    "evidence_document_ids",
    "evidence_sources",
    "evidence_titles",
    "evidence_pages",
    "evidence_sections",
    "evidence_ranks",
    "evidence_texts",
    "retrieval_strategy",
    "bm25_k",
    "vector_k",
    "top_k",
    "retrieval_query",
    "retrieval_latency_ms",
    "generation_model",
    "prompt_template_id",
    "generation_latency_ms",
    "context_tokens",
    "output_tokens",
    "condition_validity",
    "evidence_support",
    "directness",
    "meaningfulness",
    "atomicity",
    "type_validity",
    "non_redundancy",
    "condition_comment",
    "description_faithfulness",
    "description_evidence_support",
    "description_specificity",
    "unsupported_claims",
    "description_comment",
]

CONDITION_LABEL_COLUMNS = [
    "condition_validity",
    "evidence_support",
    "directness",
    "meaningfulness",
    "atomicity",
    "type_validity",
    "non_redundancy",
    "condition_comment",
]

DESCRIPTION_LABEL_COLUMNS = [
    "description_faithfulness",
    "description_evidence_support",
    "description_specificity",
    "unsupported_claims",
    "description_comment",
]

CONDITION_TYPE_ORDER = {
    "precondition": 0,
    "pre": 0,
    "postcondition": 1,
    "post": 1,
}

FIELD_TYPE_ORDER = {
    "description": 0,
    "condition": 1,
}

GROUP_COLUMNS = [
    "case_id",
    "milestone_id",
    "tactic_id",
    "attack_step_id",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create grouped annotator packages from per-case annotation_units.csv files. "
            "Attack-step groups are randomized, while description/pre/post rows remain adjacent."
        )
    )
    parser.add_argument(
        "--input-glob",
        required=True,
        help='Glob for per-case annotation_units.csv files, e.g. "evaluation_outputs/*/annotation_units.csv".',
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the annotation package is written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle attack-step groups. Default: 42.",
    )
    parser.add_argument(
        "--no-shuffle-groups",
        action="store_true",
        help="Keep attack-step groups in deterministic case/step order instead of randomizing groups.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on validation warnings. By default warnings are written to validation_report.json.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(glob.glob(args.input_glob))
    if not input_files:
        raise FileNotFoundError(f"No annotation_units.csv files matched: {args.input_glob}")

    frames: list[pd.DataFrame] = []
    validation: dict[str, Any] = {
        "input_glob": args.input_glob,
        "input_files": input_files,
        "seed": args.seed,
        "ordering": "randomized attack-step groups; description followed by preconditions and postconditions",
        "warnings": [],
        "errors": [],
    }

    for path in input_files:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        validate_required_columns(df, path, validation)
        df["source_annotation_file"] = path
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    normalize_missing_values(merged)
    reset_manual_columns(merged)
    validate_rows(merged, validation)

    if validation["errors"]:
        write_json(output_dir / "validation_report.json", validation)
        raise ValueError("Validation failed. See validation_report.json for details.")

    if args.strict and validation["warnings"]:
        write_json(output_dir / "validation_report.json", validation)
        raise ValueError("Strict validation failed due to warnings. See validation_report.json for details.")

    ordered = grouped_annotation_order(
        merged,
        seed=args.seed,
        shuffle_groups=not args.no_shuffle_groups,
    )

    # random_order is the actual order seen by annotators.  group_order lets us
    # trace which rows belong to the same attack-step group.
    ordered.insert(0, "random_order", range(1, len(ordered) + 1))

    master_path = output_dir / "annotation_master_unfilled.csv"
    ordered.to_csv(master_path, index=False)

    annotator_a = ordered.copy()
    annotator_a.insert(1, "annotator_id", "A")
    annotator_a.to_csv(output_dir / "annotator_A.csv", index=False)

    annotator_b = ordered.copy()
    annotator_b.insert(1, "annotator_id", "B")
    annotator_b.to_csv(output_dir / "annotator_B.csv", index=False)

    write_codebook(output_dir / "annotation_codebook.md")
    write_manifest(output_dir / "row_manifest.json", ordered, input_files, args.seed, not args.no_shuffle_groups)

    validation["n_rows"] = int(len(ordered))
    validation["n_description_rows"] = int((ordered["field_type"].str.lower() == "description").sum())
    validation["n_condition_rows"] = int((ordered["field_type"].str.lower() == "condition").sum())
    validation["n_attack_step_groups"] = int(ordered["annotation_group_id"].nunique())
    write_json(output_dir / "validation_report.json", validation)

    print(f"Wrote grouped annotation package to: {output_dir}")
    print(f"- {master_path}")
    print(f"- {output_dir / 'annotator_A.csv'}")
    print(f"- {output_dir / 'annotator_B.csv'}")
    print(f"- {output_dir / 'annotation_codebook.md'}")
    print(f"- {output_dir / 'row_manifest.json'}")
    print(f"- {output_dir / 'validation_report.json'}")


# ---------------------------------------------------------------------------
# Validation and normalization
# ---------------------------------------------------------------------------


def validate_required_columns(df: pd.DataFrame, path: str, validation: dict[str, Any]) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        validation["errors"].append(
            {
                "type": "missing_columns",
                "file": path,
                "missing_columns": missing,
            }
        )


def normalize_missing_values(df: pd.DataFrame) -> None:
    """Normalize common missing-value spellings to NA in-place."""
    missing_tokens = {"", "n/a", "N/A", "na", "None", "none", "null", "NULL", "-", "/"}
    for col in df.columns:
        df[col] = df[col].map(lambda x: MISSING_VALUE if str(x).strip() in missing_tokens else str(x).strip())



def reset_manual_columns(df: pd.DataFrame) -> None:
    """Clear manual-label cells and mark non-applicable labels as NA.

    This ensures annotators start from an unfilled package, even if a previous
    CSV accidentally contains labels.
    """
    field_type = df["field_type"].str.lower()
    is_description = field_type == "description"
    is_condition = field_type == "condition"

    for col in CONDITION_LABEL_COLUMNS:
        df.loc[is_condition, col] = ""
        df.loc[is_description, col] = MISSING_VALUE

    for col in DESCRIPTION_LABEL_COLUMNS:
        df.loc[is_description, col] = ""
        df.loc[is_condition, col] = MISSING_VALUE



def validate_rows(df: pd.DataFrame, validation: dict[str, Any]) -> None:
    # Unique stable IDs are essential because all downstream agreement and
    # adjudication files are aligned by annotation_unit_id.
    duplicated_ids = sorted(df.loc[df["annotation_unit_id"].duplicated(), "annotation_unit_id"].unique())
    if duplicated_ids:
        validation["errors"].append(
            {
                "type": "duplicate_annotation_unit_id",
                "count": len(duplicated_ids),
                "examples": duplicated_ids[:20],
            }
        )

    valid_field_types = {"description", "condition"}
    bad_field_type = df[~df["field_type"].str.lower().isin(valid_field_types)]
    if not bad_field_type.empty:
        validation["errors"].append(
            {
                "type": "invalid_field_type",
                "count": int(len(bad_field_type)),
                "examples": bad_field_type["annotation_unit_id"].head(20).tolist(),
            }
        )

    description_rows = df[df["field_type"].str.lower() == "description"]
    bad_description_condition_type = description_rows[
        ~description_rows["condition_type"].isin([MISSING_VALUE, ""])
    ]
    if not bad_description_condition_type.empty:
        validation["warnings"].append(
            {
                "type": "description_rows_with_condition_type",
                "count": int(len(bad_description_condition_type)),
                "examples": bad_description_condition_type["annotation_unit_id"].head(20).tolist(),
            }
        )

    condition_rows = df[df["field_type"].str.lower() == "condition"]
    bad_condition_type = condition_rows[
        ~condition_rows["condition_type"].str.lower().isin(["precondition", "postcondition", "pre", "post"])
    ]
    if not bad_condition_type.empty:
        validation["errors"].append(
            {
                "type": "condition_rows_with_missing_or_invalid_condition_type",
                "count": int(len(bad_condition_type)),
                "examples": bad_condition_type["annotation_unit_id"].head(20).tolist(),
            }
        )

    empty_generated_text = df[df["generated_field_text"].isin([MISSING_VALUE, ""])]
    if not empty_generated_text.empty:
        validation["warnings"].append(
            {
                "type": "empty_generated_field_text",
                "count": int(len(empty_generated_text)),
                "examples": empty_generated_text["annotation_unit_id"].head(20).tolist(),
            }
        )

    evidence_warnings = validate_evidence_texts(df)
    validation["warnings"].extend(evidence_warnings)



def validate_evidence_texts(df: pd.DataFrame) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []

    rows_missing_evidence_text = []
    rows_mismatched_evidence_counts = []

    for _, row in df.iterrows():
        chunk_ids = split_multi(row.get("evidence_chunk_ids", ""))
        evidence_texts = split_multi(row.get("evidence_texts", ""))

        if chunk_ids and not evidence_texts:
            rows_missing_evidence_text.append(row["annotation_unit_id"])

        if chunk_ids and evidence_texts and len(chunk_ids) != len(evidence_texts):
            rows_mismatched_evidence_counts.append(row["annotation_unit_id"])

    if rows_missing_evidence_text:
        warnings.append(
            {
                "type": "rows_with_chunk_ids_but_no_evidence_text",
                "count": len(rows_missing_evidence_text),
                "examples": rows_missing_evidence_text[:20],
            }
        )

    if rows_mismatched_evidence_counts:
        warnings.append(
            {
                "type": "rows_with_mismatched_evidence_id_and_text_counts",
                "count": len(rows_mismatched_evidence_counts),
                "examples": rows_mismatched_evidence_counts[:20],
            }
        )

    return warnings


# ---------------------------------------------------------------------------
# Grouped ordering
# ---------------------------------------------------------------------------


def grouped_annotation_order(df: pd.DataFrame, *, seed: int, shuffle_groups: bool) -> pd.DataFrame:
    """Randomize attack-step groups while keeping related rows adjacent.

    Each group corresponds to one attack step.  Inside the group, rows are sorted
    as:

        description rows
        preconditions
        postconditions

    This gives annotators the description immediately before the related
    conditions, improving local context without exposing them to the whole case
    in original order.
    """
    df = df.copy()
    df["annotation_group_id"] = df.apply(make_group_id, axis=1)
    df["field_type_sort"] = df["field_type"].str.lower().map(FIELD_TYPE_ORDER).fillna(99).astype(int)
    df["condition_type_sort"] = df["condition_type"].str.lower().map(CONDITION_TYPE_ORDER).fillna(99).astype(int)

    groups = []
    for group_id, group in df.groupby("annotation_group_id", sort=True):
        sorted_group = group.sort_values(
            by=[
                "field_type_sort",
                "condition_type_sort",
                "field_id",
                "annotation_unit_id",
            ],
            kind="stable",
        )
        groups.append((group_id, sorted_group))

    if shuffle_groups:
        rng = random.Random(seed)
        rng.shuffle(groups)

    ordered_frames = []
    for group_order, (group_id, group) in enumerate(groups, start=1):
        group = group.copy()
        group["attack_step_group_order"] = group_order
        group["row_order_within_attack_step"] = range(1, len(group) + 1)
        ordered_frames.append(group)

    ordered = pd.concat(ordered_frames, ignore_index=True)
    ordered = ordered.drop(columns=["field_type_sort", "condition_type_sort"])

    # Keep source_annotation_file for traceability, but place it at the end.
    preferred = [
        "annotation_group_id",
        "attack_step_group_order",
        "row_order_within_attack_step",
    ]
    remaining = [col for col in ordered.columns if col not in preferred]
    return ordered[preferred + remaining]



def make_group_id(row: pd.Series) -> str:
    parts = []
    for col in GROUP_COLUMNS:
        value = str(row.get(col, MISSING_VALUE)).strip()
        parts.append(value if value else MISSING_VALUE)
    return "::".join(parts)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_manifest(path: Path, df: pd.DataFrame, input_files: list[str], seed: int, shuffle_groups: bool) -> None:
    manifest = {
        "input_files": input_files,
        "seed": seed,
        "shuffle_attack_step_groups": shuffle_groups,
        "ordering_policy": {
            "between_attack_steps": "randomized by seed" if shuffle_groups else "deterministic",
            "within_attack_step": [
                "description rows",
                "precondition rows",
                "postcondition rows",
            ],
        },
        "n_rows": int(len(df)),
        "n_attack_step_groups": int(df["annotation_group_id"].nunique()),
        "n_description_rows": int((df["field_type"].str.lower() == "description").sum()),
        "n_condition_rows": int((df["field_type"].str.lower() == "condition").sum()),
        "groups": [],
    }

    for group_id, group in df.groupby("annotation_group_id", sort=False):
        manifest["groups"].append(
            {
                "annotation_group_id": group_id,
                "attack_step_group_order": int(group["attack_step_group_order"].iloc[0]),
                "case_id": str(group["case_id"].iloc[0]),
                "attack_step_id": str(group["attack_step_id"].iloc[0]),
                "attack_step_label": str(group["attack_step_label"].iloc[0]),
                "row_count": int(len(group)),
                "annotation_unit_ids": group["annotation_unit_id"].tolist(),
            }
        )

    write_json(path, manifest)



def write_codebook(path: Path) -> None:
    text = """# Annotation Codebook

## Task

You are annotating outputs produced by a CTI enrichment pipeline.  Each attack
step is shown as a small group: first the enriched description, followed by its
related preconditions and postconditions.  The grouping is intentional and is
meant to improve readability.  The attack-step groups themselves are randomized.

Annotators must work independently.  Use only the cited evidence chunks shown in
`evidence_texts` when judging evidence support.  Do not use external knowledge
unless the criterion explicitly asks about general meaningfulness or atomicity.

## Row types

- `description`: an enriched attack-step description.
- `condition`: a generated precondition or postcondition.

## Condition labels

`condition_validity`: 2 = valid, 1 = partially valid, 0 = invalid.

`evidence_support`: 2 = directly supported, 1 = inferentially supported, 0 = unsupported.

`directness`: 2 = direct, 1 = indirect but relevant, 0 = campaign-level or unrelated.

`meaningfulness`: 1 = meaningful, 0 = generic.

`atomicity`: 1 = atomic, 0 = compound.

`type_validity`: 2 = correct type, 1 = ambiguous type, 0 = wrong type.

`non_redundancy`: 2 = non-redundant, 1 = partially overlapping, 0 = redundant.

Use `condition_comment` for ambiguous or difficult cases.

## Description labels

`description_faithfulness`: 2 = faithful, 1 = partially faithful, 0 = not faithful.

`description_evidence_support`: 2 = supported, 1 = partially supported or unclear, 0 = unsupported.

`description_specificity`: 2 = specific, 1 = partially specific, 0 = generic.

`unsupported_claims`: 2 = no unsupported claims, 1 = minor unsupported claims, 0 = major unsupported claims.

Use `description_comment` for ambiguous or difficult cases.

## Not applicable fields

Cells marked `NA` are not applicable and must not be edited.
"""
    path.write_text(text, encoding="utf-8")



def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")



def split_multi(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text == MISSING_VALUE:
        return []
    return [part.strip() for part in text.split(" ||| ") if part.strip() and part.strip() != MISSING_VALUE]


if __name__ == "__main__":
    main()
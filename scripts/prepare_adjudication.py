#!/usr/bin/env python3
"""
Prepare adjudication files from two completed annotator CSV files.

This version matches rows by `annotation_unit_id` (stable row ID), not by row
position. It is robust to CSV files saved by Excel on Windows by trying multiple
encodings before failing.

Outputs:
  adjudications.csv
  adjudication_disagreements_only.csv
  agreement_input_long.csv
  adjudication_report.json

Example:
  python scripts/prepare_adjudication.py \
    --annotator-a annotation_round_1/filled/annotator_A_filled.csv \
    --annotator-b annotation_round_1/filled/annotator_B_filled.csv \
    --output-dir annotation_round_1/adjudication
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


MATCH_KEY = "annotation_unit_id"
NA_VALUE = "NA"

# Manual annotation criteria.
CONDITION_CRITERIA = [
    "condition_validity",
    "evidence_support",
    "directness",
    "meaningfulness",
    "atomicity",
    "type_validity",
    "non_redundancy",
]

DESCRIPTION_CRITERIA = [
    "description_faithfulness",
    "description_evidence_support",
    "description_specificity",
    "unsupported_claims",
]

COMMENT_COLUMNS = [
    "condition_comment",
    "description_comment",
]

ALL_CRITERIA = CONDITION_CRITERIA + DESCRIPTION_CRITERIA

# Columns copied once from annotator A to give the adjudicator context.
# Missing columns are ignored, so the script can handle small schema variations.
CONTEXT_COLUMNS = [
    "random_order",
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
]

# Values often produced by Excel/manual editing to mean "missing".
MISSING_TOKENS = {
    "",
    "na",
    "n/a",
    "n.a.",
    "none",
    "null",
    "nan",
    "-",
    "/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare adjudication files from two completed annotator CSVs."
    )
    parser.add_argument("--annotator-a", required=True, help="Filled CSV from annotator A.")
    parser.add_argument("--annotator-b", required=True, help="Filled CSV from annotator B.")
    parser.add_argument("--output-dir", required=True, help="Directory where adjudication files are written.")
    parser.add_argument(
        "--match-key",
        default=MATCH_KEY,
        help="Column used to align rows. Default: annotation_unit_id.",
    )
    parser.add_argument(
        "--keep-comments",
        action="store_true",
        help="Also include annotator comment columns as <comment>_A and <comment>_B.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    annotator_a_path = Path(args.annotator_a)
    annotator_b_path = Path(args.annotator_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a, encoding_a = load_annotation_csv(annotator_a_path)
    b, encoding_b = load_annotation_csv(annotator_b_path)

    validate_input(a, b, match_key=args.match_key)

    # Normalize manual label cells before comparison. This preserves numeric labels
    # as strings and converts common missing spellings to the canonical NA token.
    for df in (a, b):
        normalize_manual_columns(df)

    merged = a.merge(
        b,
        on=args.match_key,
        how="inner",
        suffixes=("_A_src", "_B_src"),
        validate="one_to_one",
    )

    adjudication = build_adjudication_table(
        merged=merged,
        match_key=args.match_key,
        include_comments=args.keep_comments,
    )

    disagreements = filter_disagreements(adjudication)
    agreement_long = build_agreement_input_long(adjudication)
    report = build_report(
        annotator_a_path=annotator_a_path,
        annotator_b_path=annotator_b_path,
        encoding_a=encoding_a,
        encoding_b=encoding_b,
        match_key=args.match_key,
        n_rows=len(adjudication),
        n_disagreement_rows=len(disagreements),
        adjudication=adjudication,
    )

    # utf-8-sig is intentionally used because the files are often opened in Excel
    # on Windows. It prevents accented characters from becoming mojibake.
    adjudication.to_csv(output_dir / "adjudications.csv", index=False, encoding="utf-8-sig")
    disagreements.to_csv(output_dir / "adjudication_disagreements_only.csv", index=False, encoding="utf-8-sig")
    agreement_long.to_csv(output_dir / "agreement_input_long.csv", index=False, encoding="utf-8-sig")
    (output_dir / "adjudication_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Loaded annotator A using encoding: {encoding_a}")
    print(f"Loaded annotator B using encoding: {encoding_b}")
    print(f"Matched rows using: {args.match_key}")
    print(f"Rows: {len(adjudication)}")
    print(f"Rows requiring adjudication: {len(disagreements)}")
    print(f"Wrote: {output_dir / 'adjudications.csv'}")
    print(f"Wrote: {output_dir / 'adjudication_disagreements_only.csv'}")
    print(f"Wrote: {output_dir / 'agreement_input_long.csv'}")
    print(f"Wrote: {output_dir / 'adjudication_report.json'}")


def load_annotation_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """Load a CSV with robust encoding fallback.

    The error you observed usually occurs when a CSV was saved by Excel using a
    Windows encoding such as cp1252 rather than UTF-8. We first try UTF-8 variants
    and then Windows/Latin encodings.
    """
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    errors: list[str] = []

    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
            )
            return df, encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
        except Exception as exc:
            # Keep trying only for clear parser/encoding-style issues; otherwise
            # include the message and continue because Excel-edited CSVs can be
            # mildly irregular.
            errors.append(f"{encoding}: {type(exc).__name__}: {exc}")

    # Last-resort read: replace undecodable bytes instead of failing. This should
    # rarely be needed, but it is better than losing the whole adjudication run.
    try:
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
            encoding_errors="replace",
        )
        return df, "utf-8-with-replacement"
    except Exception as exc:
        detail = "\n".join(errors)
        raise RuntimeError(f"Could not read CSV file: {path}\nTried encodings:\n{detail}") from exc


def validate_input(a: pd.DataFrame, b: pd.DataFrame, *, match_key: str) -> None:
    for name, df in [("annotator A", a), ("annotator B", b)]:
        if match_key not in df.columns:
            raise ValueError(f"{name} file is missing match key column: {match_key}")

        duplicated = df[df[match_key].duplicated()][match_key].tolist()
        if duplicated:
            sample = duplicated[:10]
            raise ValueError(f"{name} file has duplicate {match_key} values. Sample: {sample}")

        missing_criteria = [c for c in ALL_CRITERIA if c not in df.columns]
        if missing_criteria:
            raise ValueError(f"{name} file is missing annotation columns: {missing_criteria}")

    ids_a = set(a[match_key].astype(str))
    ids_b = set(b[match_key].astype(str))

    only_a = sorted(ids_a - ids_b)
    only_b = sorted(ids_b - ids_a)

    if only_a or only_b:
        raise ValueError(
            f"Annotator files do not contain the same {match_key} values.\n"
            f"Only in A: {only_a[:10]}\n"
            f"Only in B: {only_b[:10]}"
        )


def normalize_manual_columns(df: pd.DataFrame) -> None:
    for col in ALL_CRITERIA + COMMENT_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df[col].map(normalize_cell)


def normalize_cell(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return NA_VALUE
    # Excel sometimes stores labels as 1.0 or 2.0. Normalize them back to 1/2.
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def merged_value(row: pd.Series, base_col: str, side: str, *, match_key: str) -> str:
    """Get a column value after pandas merge with suffixes.

    The match key appears only once. Other shared columns are suffixed.
    """
    if base_col == match_key:
        return str(row.get(match_key, ""))

    suffixed = f"{base_col}_{side}_src"
    if suffixed in row.index:
        return str(row.get(suffixed, ""))

    # If the column existed in only one dataframe, pandas may keep it unsuffixed.
    if base_col in row.index:
        return str(row.get(base_col, ""))

    return ""


def build_adjudication_table(
    *,
    merged: pd.DataFrame,
    match_key: str,
    include_comments: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        out: dict[str, Any] = {}

        # Copy context from annotator A. The generated text and evidence are
        # expected to be identical across both files.
        for col in CONTEXT_COLUMNS:
            if col == match_key or f"{col}_A_src" in row.index or col in row.index:
                out[col] = merged_value(row, col, "A", match_key=match_key)

        # Preserve annotator IDs if present.
        out["annotator_id_A"] = merged_value(row, "annotator_id", "A", match_key=match_key)
        out["annotator_id_B"] = merged_value(row, "annotator_id", "B", match_key=match_key)

        row_has_disagreement = False

        for criterion in ALL_CRITERIA:
            value_a = merged_value(row, criterion, "A", match_key=match_key)
            value_b = merged_value(row, criterion, "B", match_key=match_key)

            value_a = normalize_cell(value_a)
            value_b = normalize_cell(value_b)

            agree = value_a == value_b
            applicable = is_criterion_applicable(out.get("field_type", ""), criterion)

            out[f"{criterion}_A"] = value_a
            out[f"{criterion}_B"] = value_b
            out[f"{criterion}_agree"] = "true" if agree else "false"

            # If a criterion is not applicable, keep final as NA. If it is
            # applicable and annotators agree, pre-fill. Otherwise leave empty
            # for adjudication.
            if not applicable:
                out[f"final_{criterion}"] = NA_VALUE
            elif agree:
                out[f"final_{criterion}"] = value_a
            else:
                out[f"final_{criterion}"] = ""
                row_has_disagreement = True

            out[f"{criterion}_adjudication_comment"] = ""

        if include_comments:
            for comment_col in COMMENT_COLUMNS:
                out[f"{comment_col}_A"] = merged_value(row, comment_col, "A", match_key=match_key)
                out[f"{comment_col}_B"] = merged_value(row, comment_col, "B", match_key=match_key)

        out["requires_adjudication"] = "true" if row_has_disagreement else "false"
        rows.append(out)

    return pd.DataFrame(rows)


def is_criterion_applicable(field_type: str, criterion: str) -> bool:
    field_type = str(field_type).strip().lower()
    if criterion in CONDITION_CRITERIA:
        return field_type == "condition"
    if criterion in DESCRIPTION_CRITERIA:
        return field_type == "description"
    return False


def filter_disagreements(adjudication: pd.DataFrame) -> pd.DataFrame:
    if "requires_adjudication" not in adjudication.columns:
        return adjudication.iloc[0:0].copy()
    return adjudication[adjudication["requires_adjudication"] == "true"].copy()


def build_agreement_input_long(adjudication: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in adjudication.iterrows():
        field_type = str(row.get("field_type", ""))
        for criterion in ALL_CRITERIA:
            if not is_criterion_applicable(field_type, criterion):
                continue
            rows.append(
                {
                    "annotation_unit_id": row.get("annotation_unit_id", ""),
                    "random_order": row.get("random_order", ""),
                    "case_id": row.get("case_id", ""),
                    "field_type": field_type,
                    "criterion": criterion,
                    "label_A": row.get(f"{criterion}_A", ""),
                    "label_B": row.get(f"{criterion}_B", ""),
                    "agree": row.get(f"{criterion}_agree", ""),
                }
            )

    return pd.DataFrame(rows)


def build_report(
    *,
    annotator_a_path: Path,
    annotator_b_path: Path,
    encoding_a: str,
    encoding_b: str,
    match_key: str,
    n_rows: int,
    n_disagreement_rows: int,
    adjudication: pd.DataFrame,
) -> dict[str, Any]:
    per_criterion: dict[str, Any] = {}
    for criterion in ALL_CRITERIA:
        agree_col = f"{criterion}_agree"
        if agree_col not in adjudication.columns:
            continue
        applicable_mask = adjudication.apply(
            lambda row: is_criterion_applicable(row.get("field_type", ""), criterion),
            axis=1,
        )
        applicable = adjudication[applicable_mask]
        n = len(applicable)
        if n == 0:
            raw_agreement = None
            disagreements = 0
        else:
            agreements = int((applicable[agree_col] == "true").sum())
            disagreements = n - agreements
            raw_agreement = agreements / n
        per_criterion[criterion] = {
            "n": n,
            "raw_agreement": raw_agreement,
            "disagreements": disagreements,
        }

    return {
        "inputs": {
            "annotator_a": str(annotator_a_path),
            "annotator_b": str(annotator_b_path),
            "encoding_a": encoding_a,
            "encoding_b": encoding_b,
            "match_key": match_key,
        },
        "rows": {
            "total": n_rows,
            "requiring_adjudication": n_disagreement_rows,
        },
        "criteria": per_criterion,
    }


if __name__ == "__main__":
    main()

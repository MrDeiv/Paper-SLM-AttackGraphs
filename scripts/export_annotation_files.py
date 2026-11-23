#!/usr/bin/env python3
"""
Export annotation-ready CSV files from experiment JSONL outputs.

This script reads one or more outputs/runs/<run_id>/ directories and creates
CSV files for human annotation:

- conditions_to_annotate.csv
- descriptions_to_annotate.csv
- retrieval_to_annotate.csv

It writes both per-run exports under each run directory and combined exports in
the requested output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.common import flatten_list_field, write_csv_rows
from src.io_utils import read_jsonl


CONDITION_FIELDS = [
    "run_id",
    "case_id",
    "attack_step_id",
    "condition_id",
    "condition_type",
    "category",
    "statement",
    "evidence_chunk_ids",
    "cited_chunk_text",
    "source_documents",
    "system_support_level",
    "system_directness",
    "system_status",
    "verifier_rationale",
    "annotator_id",
    "annotator_valid",
    "annotator_grounded",
    "annotator_directness",
    "annotator_redundant",
    "annotator_category_correct",
    "annotator_notes",
]

DESCRIPTION_FIELDS = [
    "run_id",
    "case_id",
    "attack_step_id",
    "description",
    "evidence_chunk_ids",
    "cited_chunk_text",
    "source_documents",
    "system_support_level",
    "system_status",
    "annotator_id",
    "annotator_faithfulness",
    "annotator_specificity",
    "annotator_grounded",
    "annotator_notes",
]

RETRIEVAL_FIELDS = [
    "run_id",
    "case_id",
    "attack_step_id",
    "retriever",
    "chunking_strategy",
    "rank",
    "chunk_id",
    "document_id",
    "source_path",
    "chunk_text",
    "query",
    "annotator_id",
    "annotator_relevant",
    "annotator_evidence_type",
    "annotator_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export annotation-ready CSV files from run outputs.")
    parser.add_argument("--runs", default="outputs/runs", help="Directory containing run output folders.")
    parser.add_argument("--run", action="append", help="Specific run_id to export. Can be provided multiple times.")
    parser.add_argument("--out", default="data/annotation", help="Directory for combined annotation exports.")
    parser.add_argument(
        "--conditions-source",
        choices=["verified", "generated", "final"],
        default="verified",
        help="Which condition file to export for annotation.",
    )
    parser.add_argument("--max-chunk-chars", type=int, default=1600, help="Maximum cited chunk text length per chunk.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_run_dirs(runs_root, args.run)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_root}")

    all_condition_rows: list[dict[str, Any]] = []
    all_description_rows: list[dict[str, Any]] = []
    all_retrieval_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        chunk_map = _load_chunk_map(run_dir)
        condition_rows = build_condition_rows(run_dir, chunk_map, args.conditions_source, args.max_chunk_chars)
        description_rows = build_description_rows(run_dir, chunk_map, args.max_chunk_chars)
        retrieval_rows = build_retrieval_rows(run_dir, args.max_chunk_chars)

        per_run_out = run_dir / "annotation_exports"
        per_run_out.mkdir(parents=True, exist_ok=True)
        write_csv_rows(per_run_out / "conditions_to_annotate.csv", condition_rows, CONDITION_FIELDS)
        write_csv_rows(per_run_out / "descriptions_to_annotate.csv", description_rows, DESCRIPTION_FIELDS)
        write_csv_rows(per_run_out / "retrieval_to_annotate.csv", retrieval_rows, RETRIEVAL_FIELDS)

        all_condition_rows.extend(condition_rows)
        all_description_rows.extend(description_rows)
        all_retrieval_rows.extend(retrieval_rows)

    write_csv_rows(out_dir / "conditions_to_annotate.csv", all_condition_rows, CONDITION_FIELDS)
    write_csv_rows(out_dir / "descriptions_to_annotate.csv", all_description_rows, DESCRIPTION_FIELDS)
    write_csv_rows(out_dir / "retrieval_to_annotate.csv", all_retrieval_rows, RETRIEVAL_FIELDS)

    print(f"Exported condition rows: {len(all_condition_rows)}")
    print(f"Exported description rows: {len(all_description_rows)}")
    print(f"Exported retrieval rows: {len(all_retrieval_rows)}")
    print(f"Combined exports: {out_dir}")


def build_condition_rows(
    run_dir: Path,
    chunk_map: dict[str, dict[str, Any]],
    source: str,
    max_chunk_chars: int,
) -> list[dict[str, Any]]:
    filename = {
        "verified": "verified_conditions.jsonl",
        "generated": "generated_conditions.jsonl",
        "final": "final_conditions.jsonl",
    }[source]
    conditions = read_jsonl(run_dir / filename)
    rows: list[dict[str, Any]] = []

    for condition in conditions:
        evidence_ids = _as_list(condition.get("evidence_chunk_ids"))
        rows.append(
            {
                "run_id": condition.get("run_id", run_dir.name),
                "case_id": condition.get("case_id", ""),
                "attack_step_id": condition.get("attack_step_id", ""),
                "condition_id": condition.get("condition_id", ""),
                "condition_type": condition.get("condition_type", ""),
                "category": condition.get("category", ""),
                "statement": condition.get("statement", ""),
                "evidence_chunk_ids": flatten_list_field(evidence_ids),
                "cited_chunk_text": _joined_chunk_text(evidence_ids, chunk_map, max_chunk_chars),
                "source_documents": flatten_list_field(condition.get("source_documents", [])),
                "system_support_level": condition.get("support_level", ""),
                "system_directness": condition.get("directness", ""),
                "system_status": condition.get("status", ""),
                "verifier_rationale": condition.get("verifier_rationale", ""),
                "annotator_id": "",
                "annotator_valid": "",
                "annotator_grounded": "",
                "annotator_directness": "",
                "annotator_redundant": "",
                "annotator_category_correct": "",
                "annotator_notes": "",
            }
        )

    return rows


def build_description_rows(
    run_dir: Path,
    chunk_map: dict[str, dict[str, Any]],
    max_chunk_chars: int,
) -> list[dict[str, Any]]:
    descriptions = read_jsonl(run_dir / "generated_descriptions.jsonl")
    rows: list[dict[str, Any]] = []

    for description in descriptions:
        evidence_ids = _as_list(description.get("evidence_chunk_ids"))
        rows.append(
            {
                "run_id": description.get("run_id", run_dir.name),
                "case_id": description.get("case_id", ""),
                "attack_step_id": description.get("attack_step_id", ""),
                "description": description.get("description", ""),
                "evidence_chunk_ids": flatten_list_field(evidence_ids),
                "cited_chunk_text": _joined_chunk_text(evidence_ids, chunk_map, max_chunk_chars),
                "source_documents": flatten_list_field(description.get("source_documents", [])),
                "system_support_level": description.get("support_level", ""),
                "system_status": description.get("status", ""),
                "annotator_id": "",
                "annotator_faithfulness": "",
                "annotator_specificity": "",
                "annotator_grounded": "",
                "annotator_notes": "",
            }
        )

    return rows


def build_retrieval_rows(run_dir: Path, max_chunk_chars: int) -> list[dict[str, Any]]:
    retrieval_records = read_jsonl(run_dir / "retrieval_results.jsonl")
    rows: list[dict[str, Any]] = []

    for record in retrieval_records:
        for result in record.get("results", []):
            rows.append(
                {
                    "run_id": record.get("run_id", run_dir.name),
                    "case_id": record.get("case_id", ""),
                    "attack_step_id": record.get("attack_step_id", ""),
                    "retriever": record.get("retriever", ""),
                    "chunking_strategy": record.get("chunking_strategy", ""),
                    "rank": result.get("rank", ""),
                    "chunk_id": result.get("chunk_id", ""),
                    "document_id": result.get("document_id", ""),
                    "source_path": result.get("source_path", ""),
                    "chunk_text": str(result.get("text", ""))[:max_chunk_chars],
                    "query": record.get("query", ""),
                    "annotator_id": "",
                    "annotator_relevant": "",
                    "annotator_evidence_type": "",
                    "annotator_notes": "",
                }
            )

    return rows


def _discover_run_dirs(runs_root: Path, selected_runs: list[str] | None) -> list[Path]:
    if selected_runs:
        return [runs_root / run_id for run_id in selected_runs if (runs_root / run_id).exists()]
    return sorted(path for path in runs_root.iterdir() if path.is_dir() and (path / "config.json").exists())


def _load_chunk_map(run_dir: Path) -> dict[str, dict[str, Any]]:
    chunks = read_jsonl(run_dir / "chunks.jsonl")
    return {str(chunk.get("chunk_id", "")): chunk for chunk in chunks if chunk.get("chunk_id")}


def _joined_chunk_text(ids: list[str], chunk_map: dict[str, dict[str, Any]], max_chunk_chars: int) -> str:
    blocks: list[str] = []
    for chunk_id in ids:
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue
        source = chunk.get("source_path", "")
        text = str(chunk.get("text", ""))[:max_chunk_chars]
        blocks.append(f"[chunk_id: {chunk_id}]\n[source: {source}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        if ";" in value:
            return [part.strip() for part in value.split(";") if part.strip()]
        return [value] if value.strip() else []
    return [str(value)]


if __name__ == "__main__":
    main()
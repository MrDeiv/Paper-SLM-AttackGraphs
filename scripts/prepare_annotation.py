#!/usr/bin/env python3
"""Prepare annotation_units.csv files from final pipeline outputs.

This version is intentionally stricter than the generic annotation-unit builder:
it reads conditions only from `final_conditions.jsonl`. It does not fall back to
`generated_conditions.jsonl`, because annotators should evaluate the final
post-processed output of the pipeline, not intermediate generated conditions
that may include duplicates removed by the semantic duplicate filter.

Expected app3.py output layout:

    output/<case-id>/
      config_snapshot.json
      enriched_attack_graph.json
      generated_descriptions.jsonl
      final_conditions.jsonl        # REQUIRED condition source
      retrieval_logs.jsonl
      generation_logs.jsonl
      chunks.jsonl
      metrics.json

It creates one annotation_units.csv per case, with one row per enriched
attack-step description and one row per final generated precondition/postcondition.
The output schema follows the manual-evaluation protocol used in the paper.

Examples
--------
Prepare annotation units for all cases under output/:

    python scripts/prepare_annotation_units_final.py \
      --input-root output \
      --output-root evaluation_outputs

Prepare only one case:

    python scripts/prepare_annotation_units_final.py \
      --input-root output \
      --output-root evaluation_outputs \
      --case goofy

Fail if optional evidence inconsistencies are detected:

    python scripts/prepare_annotation_units_final.py \
      --input-root output \
      --output-root evaluation_outputs \
      --strict
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

SAFE_DELIMITER = " ||| "
NA = "NA"

# Keep this header aligned with the manual evaluation protocol.
ANNOTATION_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create annotation_units.csv files from app3.py final_conditions.jsonl outputs."
    )
    parser.add_argument(
        "--input-root",
        default="output",
        help="Root directory containing output/<case-id>/ folders. Default: output",
    )
    parser.add_argument(
        "--output-root",
        default="evaluation_outputs",
        help="Root directory where evaluation_outputs/<case-id>/annotation_units.csv is written.",
    )
    parser.add_argument(
        "--case",
        default="all",
        help="Case ID to process, or 'all'. Default: all",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing optional files or evidence inconsistencies instead of warning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    case_dirs = discover_case_dirs(input_root, args.case)
    if not case_dirs:
        raise FileNotFoundError(f"No case directories found under {input_root} for case={args.case!r}")

    manifest: list[dict[str, Any]] = []
    total_rows = 0

    for case_dir in case_dirs:
        case_id = case_dir.name
        print(f"[+] Preparing annotation units for case: {case_id}")
        rows, report = build_case_annotation_rows(case_id, case_dir, strict=args.strict)

        out_dir = output_root / case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "annotation_units.csv"
        report_path = out_dir / "annotation_units_report.json"

        write_csv(csv_path, rows, ANNOTATION_COLUMNS)
        write_json(report_path, report)

        print(f"    rows: {len(rows)}")
        print(f"    wrote: {csv_path}")
        total_rows += len(rows)
        manifest.append(
            {
                "case_id": case_id,
                "input_dir": str(case_dir),
                "output_csv": str(csv_path),
                "rows": len(rows),
                "descriptions": report["counts"]["description_rows"],
                "conditions": report["counts"]["condition_rows"],
                "warnings": len(report["warnings"]),
            }
        )

    write_json(output_root / "annotation_units_manifest.json", manifest)
    print(f"\nDone. Cases: {len(case_dirs)}. Total annotation rows: {total_rows}.")
    print(f"Manifest: {output_root / 'annotation_units_manifest.json'}")


def discover_case_dirs(input_root: Path, requested_case: str) -> list[Path]:
    if requested_case.lower() != "all":
        path = input_root / requested_case
        return [path] if path.exists() and path.is_dir() else []

    dirs = []
    for path in sorted(input_root.iterdir() if input_root.exists() else []):
        if path.is_dir() and (path / "metrics.json").exists():
            dirs.append(path)
    return dirs


def build_case_annotation_rows(case_id: str, case_dir: Path, *, strict: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    warnings: list[str] = []

    config = read_json_optional(case_dir / "config_snapshot.json", default={})
    metrics = read_json_optional(case_dir / "metrics.json", default={})
    graph = read_json_optional(case_dir / "enriched_attack_graph.json", default=[])
    if not graph:
        # app3 may write a custom OUTPUT_FILE name. Fall back to the config value.
        output_file = config.get("OUTPUT_FILE") or "enriched_attack_graph.json"
        graph = read_json_optional(case_dir / str(output_file), default=[])

    descriptions = read_jsonl_optional(case_dir / "generated_descriptions.jsonl", default=[])
    # Final condition source is mandatory in this version. Do not fall back to
    # generated_conditions.jsonl, otherwise removed semantic duplicates may be
    # shown to annotators.
    conditions_path = case_dir / "final_conditions.jsonl"
    if not conditions_path.exists():
        raise FileNotFoundError(
            f"{case_id}: required final condition file not found: {conditions_path}. "
            "Run app3.py first and make sure duplicate filtering writes final_conditions.jsonl."
        )
    conditions = read_jsonl_optional(conditions_path, default=[])
    retrieval_logs = read_jsonl_optional(case_dir / "retrieval_logs.jsonl", default=[])
    generation_logs = read_jsonl_optional(case_dir / "generation_logs.jsonl", default=[])
    chunks = read_jsonl_optional(case_dir / "chunks.jsonl", default=[])

    required = {
        "generated_descriptions.jsonl": descriptions,
        "final_conditions.jsonl": conditions,
        "chunks.jsonl": chunks,
    }
    for name, value in required.items():
        if not value:
            msg = f"{case_id}: {name} is missing or empty"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

    chunk_index = build_chunk_index(chunks)
    step_index = build_step_index(graph)
    descriptions_by_step = {str(d.get("attack_step_id")): d for d in descriptions}
    retrieval_index = build_log_index(retrieval_logs, key_field="field_type")
    generation_index = build_log_index(generation_logs, key_field="field_type")

    run_id = infer_run_id(config, metrics, case_dir)
    case_name = infer_case_name(config, case_id)
    rag = infer_rag_config(config, metrics)
    generation_model = infer_generation_model(config, metrics)

    rows: list[dict[str, Any]] = []

    for desc in descriptions:
        step_id = str(desc.get("attack_step_id") or "")
        step = step_index.get(step_id, {})
        retrieval = lookup_log(retrieval_index, step_id, "description")
        generation = lookup_log(generation_index, step_id, "description")

        row = base_row(
            run_id=run_id,
            case_id=case_id,
            case_name=case_name,
            step=step,
            field_type="description",
            field_id=str(desc.get("field_id") or f"{step_id}-desc01"),
            condition_type=NA,
        )
        generated_text = clean_text(desc.get("generated_text") or desc.get("description") or step.get("description") or "")
        row.update(
            {
                "annotation_unit_id": f"{case_id}::{step_id}::description::{row['field_id']}",
                "original_description": clean_text(desc.get("original_description") or step.get("original_description") or ""),
                "enriched_description": generated_text,
                "generated_field_text": generated_text,
                **evidence_columns(desc, chunk_index, warnings, strict, case_id, row["annotation_unit_id"]),
                **retrieval_generation_columns(rag, generation_model, "description_enrichment_v1", retrieval, generation),
            }
        )

        # Description rows: condition criteria are not applicable; description criteria are left blank for annotation.
        for col in CONDITION_LABEL_COLUMNS:
            row[col] = NA
        for col in DESCRIPTION_LABEL_COLUMNS:
            row[col] = ""

        rows.append(normalize_row(row))

    for cond in conditions:
        step_id = str(cond.get("attack_step_id") or "")
        step = step_index.get(step_id, {})
        condition_type = normalize_condition_type(cond.get("condition_type"))
        field_type_for_logs = "preconditions" if condition_type == "precondition" else "postconditions"
        retrieval = lookup_log(retrieval_index, step_id, field_type_for_logs)
        generation = lookup_log(generation_index, step_id, field_type_for_logs)
        desc = descriptions_by_step.get(step_id, {})

        field_id = str(cond.get("condition_id") or cond.get("field_id") or make_condition_id(case_id, step_id, condition_type, len(rows)))
        row = base_row(
            run_id=run_id,
            case_id=case_id,
            case_name=case_name,
            step=step,
            field_type="condition",
            field_id=field_id,
            condition_type=condition_type,
        )
        generated_text = clean_text(cond.get("statement") or cond.get("generated_text") or "")
        enriched_description = clean_text(
            desc.get("generated_text")
            or desc.get("description")
            or step.get("description")
            or step.get("enriched_description")
            or ""
        )
        row.update(
            {
                "annotation_unit_id": f"{case_id}::{step_id}::condition::{field_id}",
                "original_description": clean_text(step.get("original_description") or desc.get("original_description") or ""),
                "enriched_description": enriched_description,
                "generated_field_text": generated_text,
                **evidence_columns(cond, chunk_index, warnings, strict, case_id, f"{case_id}::{step_id}::condition::{field_id}"),
                **retrieval_generation_columns(rag, generation_model, f"{field_type_for_logs}_generation_v1", retrieval, generation),
            }
        )

        # Condition rows: condition criteria are left blank for annotation; description criteria are not applicable.
        for col in CONDITION_LABEL_COLUMNS:
            row[col] = ""
        for col in DESCRIPTION_LABEL_COLUMNS:
            row[col] = NA

        rows.append(normalize_row(row))

    validate_rows(rows, warnings, strict)

    report = {
        "case_id": case_id,
        "input_dir": str(case_dir),
        "condition_source": "final_conditions.jsonl",
        "counts": {
            "description_rows": sum(1 for r in rows if r["field_type"] == "description"),
            "condition_rows": sum(1 for r in rows if r["field_type"] == "condition"),
            "total_rows": len(rows),
            "chunks_indexed": len(chunk_index),
            "retrieval_logs": len(retrieval_logs),
            "generation_logs": len(generation_logs),
        },
        "warnings": warnings,
    }
    return rows, report


def base_row(
    *,
    run_id: str,
    case_id: str,
    case_name: str,
    step: dict[str, Any],
    field_type: str,
    field_id: str,
    condition_type: str,
) -> dict[str, Any]:
    technique = step.get("mitre_technique") if isinstance(step.get("mitre_technique"), dict) else {}
    attack_step_id = str(step.get("id") or step.get("attack_step_id") or "")
    tactic = step.get("tactic") or step.get("milestone_label") or ""
    milestone_id = str(step.get("milestone_id") or "")
    milestone_label = str(step.get("milestone_label") or tactic or "")

    return {
        "annotation_unit_id": "",
        "run_id": run_id,
        "case_id": case_id,
        "case_name": case_name,
        "milestone_id": milestone_id,
        "milestone_label": milestone_label,
        "tactic_id": slugify(tactic),
        "tactic_name": str(tactic or ""),
        "attack_step_id": attack_step_id,
        "attack_step_label": clean_text(step.get("name") or step.get("attack_step_label") or ""),
        "technique_id": clean_text(technique.get("id") or step.get("technique_id") or ""),
        "technique_name": clean_text(technique.get("name") or step.get("technique_name") or ""),
        "field_type": field_type,
        "field_id": field_id,
        "condition_type": condition_type,
    }


def build_step_index(graph: Any) -> dict[str, dict[str, Any]]:
    """Index attack steps from app3 enriched_attack_graph.json output."""
    index: dict[str, dict[str, Any]] = {}
    if not isinstance(graph, list):
        return index

    for milestone in graph:
        if not isinstance(milestone, dict):
            continue
        milestone_id = str(milestone.get("id") or milestone.get("milestone_id") or "")
        milestone_label = str(milestone.get("tactic") or milestone.get("label") or milestone.get("milestone_label") or "")
        for step in milestone.get("attack_steps", []) or []:
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("id") or step.get("attack_step_id") or "")
            if not step_id:
                continue
            merged = dict(step)
            merged.setdefault("milestone_id", milestone_id)
            merged.setdefault("milestone_label", milestone_label)
            merged.setdefault("tactic", step.get("tactic") or milestone_label)
            index[step_id] = merged
    return index


def build_chunk_index(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id") or "")
        if chunk_id:
            index[chunk_id] = chunk
    return index


def build_log_index(logs: list[dict[str, Any]], *, key_field: str) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in logs:
        step_id = str(row.get("attack_step_id") or "")
        field_type = str(row.get(key_field) or "")
        if step_id and field_type:
            index[(step_id, field_type)] = row
    return index


def lookup_log(index: dict[tuple[str, str], dict[str, Any]], attack_step_id: str, field_type: str) -> dict[str, Any]:
    return index.get((str(attack_step_id), field_type), {})


def evidence_columns(
    record: dict[str, Any],
    chunk_index: dict[str, dict[str, Any]],
    warnings: list[str],
    strict: bool,
    case_id: str,
    annotation_unit_id: str,
) -> dict[str, str]:
    """Build evidence columns and fill missing metadata from chunks.jsonl when possible."""
    chunk_ids = to_list(record.get("evidence_chunk_ids"))
    evidence_texts = to_list(record.get("evidence_texts"))
    document_ids = to_list(record.get("evidence_document_ids"))
    sources = to_list(record.get("evidence_sources"))
    titles = to_list(record.get("evidence_titles"))
    ranks = to_list(record.get("evidence_ranks"))

    pages: list[str] = []
    sections: list[str] = []

    # Fill evidence metadata from chunks.jsonl to make annotation rows self-contained.
    for i, cid in enumerate(chunk_ids):
        chunk = chunk_index.get(cid, {})
        if not chunk:
            msg = f"{case_id}: evidence chunk not found for {annotation_unit_id}: {cid}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

        if i >= len(evidence_texts) or not evidence_texts[i]:
            evidence_texts.append(str(chunk.get("text") or chunk.get("page_content") or ""))
        if i >= len(document_ids) or not document_ids[i]:
            document_ids.append(str(chunk.get("document_id") or ""))
        if i >= len(sources) or not sources[i]:
            sources.append(str(chunk.get("source") or ""))
        if i >= len(titles) or not titles[i]:
            titles.append(str(chunk.get("title") or ""))
        pages.append(str(chunk.get("page") or chunk.get("page_number") or NA))
        sections.append(str(chunk.get("section") or chunk.get("section_title") or NA))

    # Align lists to chunk count.
    n = len(chunk_ids)
    evidence_texts = pad_to(evidence_texts, n, "")
    document_ids = pad_to(document_ids, n, "")
    sources = pad_to(sources, n, "")
    titles = pad_to(titles, n, "")
    ranks = pad_to(ranks, n, "") or [str(i + 1) for i in range(n)]
    pages = pad_to(pages, n, NA)
    sections = pad_to(sections, n, NA)

    if chunk_ids and not any(clean_text(t) for t in evidence_texts):
        msg = f"{case_id}: cited evidence has no text for {annotation_unit_id}"
        if strict:
            raise ValueError(msg)
        warnings.append(msg)

    return {
        "evidence_chunk_ids": join_values(chunk_ids),
        "evidence_document_ids": join_values(document_ids),
        "evidence_sources": join_values(sources),
        "evidence_titles": join_values(titles),
        "evidence_pages": join_values(pages),
        "evidence_sections": join_values(sections),
        "evidence_ranks": join_values(ranks),
        "evidence_texts": join_values(evidence_texts),
    }


def retrieval_generation_columns(
    rag: dict[str, Any],
    generation_model: str,
    prompt_template_id: str,
    retrieval: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "retrieval_strategy": rag.get("strategy", "hybrid"),
        "bm25_k": rag.get("bm25_k", ""),
        "vector_k": rag.get("vector_k", ""),
        "top_k": rag.get("top_k", ""),
        "retrieval_query": clean_text(retrieval.get("query") or ""),
        "retrieval_latency_ms": retrieval.get("latency_ms", ""),
        "generation_model": generation_model,
        "prompt_template_id": prompt_template_id,
        "generation_latency_ms": generation.get("latency_ms", ""),
        "context_tokens": generation.get("context_tokens", ""),
        "output_tokens": generation.get("output_tokens", ""),
    }


def infer_run_id(config: dict[str, Any], metrics: dict[str, Any], case_dir: Path) -> str:
    return str(
        config.get("RUN_ID")
        or config.get("run_id")
        or config.get("CASE_ID")
        or metrics.get("run_id")
        or case_dir.name
    )


def infer_case_name(config: dict[str, Any], case_id: str) -> str:
    return str(config.get("CASE_NAME") or config.get("case_name") or case_id.replace("_", " ").replace("-", " ").title())


def infer_rag_config(config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    rag = {}
    if isinstance(config.get("RAG"), dict):
        rag.update(config["RAG"])
    metric_rag = metrics.get("pipeline_config", {}).get("rag", {}) if isinstance(metrics.get("pipeline_config"), dict) else {}
    if isinstance(metric_rag, dict):
        rag = {**metric_rag, **rag}

    # Legacy keys used by earlier scripts.
    rag.setdefault("strategy", "hybrid")
    rag.setdefault("bm25_k", config.get("BM25_k", ""))
    rag.setdefault("vector_k", config.get("k", ""))
    rag.setdefault("top_k", config.get("TOP_K", ""))
    rag.setdefault("hybrid_alpha", config.get("HYBRID_ALPHA", ""))
    return rag


def infer_generation_model(config: dict[str, Any], metrics: dict[str, Any]) -> str:
    models = config.get("MODELS", {}) if isinstance(config.get("MODELS"), dict) else {}
    if models.get("TEXT_GENERATION"):
        return str(models["TEXT_GENERATION"])
    metric_models = metrics.get("pipeline_config", {}).get("models", {}) if isinstance(metrics.get("pipeline_config"), dict) else {}
    if isinstance(metric_models, dict) and metric_models.get("TEXT_GENERATION"):
        return str(metric_models["TEXT_GENERATION"])
    return ""


def validate_rows(rows: list[dict[str, Any]], warnings: list[str], strict: bool) -> None:
    seen: set[str] = set()
    for row in rows:
        uid = row.get("annotation_unit_id", "")
        if not uid:
            raise ValueError("annotation row without annotation_unit_id")
        if uid in seen:
            raise ValueError(f"duplicate annotation_unit_id: {uid}")
        seen.add(uid)

        if row.get("field_type") == "condition" and row.get("condition_type") not in {"precondition", "postcondition"}:
            raise ValueError(f"condition row with invalid condition_type: {uid}")
        if row.get("field_type") == "description" and row.get("condition_type") not in {NA, "", None}:
            raise ValueError(f"description row has non-empty condition_type: {uid}")
        if not clean_text(row.get("generated_field_text")):
            msg = f"empty generated_field_text: {uid}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)


def normalize_row(row: dict[str, Any]) -> dict[str, str]:
    out = {}
    for col in ANNOTATION_COLUMNS:
        value = row.get(col, "")
        if value is None:
            value = ""
        if isinstance(value, (list, tuple, set)):
            value = join_values(list(value))
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        out[col] = str(value)
    return out


def normalize_condition_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"precondition", "pre-condition", "pre"}:
        return "precondition"
    if text in {"postcondition", "post-condition", "post"}:
        return "postcondition"
    return text or "condition"


def make_condition_id(case_id: str, step_id: str, condition_type: str, n: int) -> str:
    prefix = "pre" if condition_type == "precondition" else "post"
    return f"{step_id or case_id}-{prefix}{n:04d}"


def read_json_optional(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl_optional(path: Path, default: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return default
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    if isinstance(value, str):
        if not value:
            return []
        if SAFE_DELIMITER in value:
            return [v.strip() for v in value.split(SAFE_DELIMITER)]
        return [value]
    return [str(value)]


def pad_to(values: list[str], n: int, fill: str) -> list[str]:
    out = list(values[:n])
    while len(out) < n:
        out.append(fill)
    return out


def join_values(values: Iterable[Any]) -> str:
    return SAFE_DELIMITER.join(clean_text(v) for v in values)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def slugify(value: Any) -> str:
    text = str(value or "unknown").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


if __name__ == "__main__":
    main()
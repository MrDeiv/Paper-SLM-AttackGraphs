#!/usr/bin/env python3
"""Evaluate one completed CTI-enrichment pipeline case.

This script is intentionally standalone and artifact-driven. It does not run the
pipeline, does not call the RAG retriever, and does not call the LLM. It reads
one completed case directory and produces the two per-case evaluation artifacts
required by the manual-evaluation workflow:

    1. summary.json
    2. annotation_units.csv

Expected usage:

    python scripts/evaluate_case.py \
      --case goofy \
      --input-dir case_outputs/goofy \
      --output-dir evaluation_outputs/goofy

The script is designed for the simplified no-verifier pipeline:

    - evidence is deterministically attached from retrieved RAG chunks;
    - generated descriptions and generated conditions already contain or inherit
      evidence_chunk_ids;
    - final_conditions.jsonl is used when present, otherwise generated_conditions.jsonl;
    - no verifier support labels are required.

It is also deliberately permissive about artifact schemas. The production
pipeline evolved during the project, so this evaluator accepts several common
field aliases while preserving the CSV schema required by the paper protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

SAFE_DELIM = " ||| "
NA = "NA"

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

CONDITION_MANUAL_COLUMNS = [
    "condition_validity",
    "evidence_support",
    "directness",
    "meaningfulness",
    "atomicity",
    "type_validity",
    "non_redundancy",
    "condition_comment",
]

DESCRIPTION_MANUAL_COLUMNS = [
    "description_faithfulness",
    "description_evidence_support",
    "description_specificity",
    "unsupported_claims",
    "description_comment",
]


@dataclass
class RetrievalMetadata:
    """Step-level retrieval metadata used in annotation rows."""

    query: str = ""
    latency_ms: Any = ""
    retrieved_chunk_ids: list[str] | None = None
    chunk_rank: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.retrieved_chunk_ids is None:
            self.retrieved_chunk_ids = []
        if self.chunk_rank is None:
            self.chunk_rank = {}


@dataclass
class GenerationMetadata:
    """Step-level generation metadata used in annotation rows."""

    model: str = ""
    prompt_template_id: str = ""
    latency_ms: Any = ""
    context_tokens: Any = ""
    output_tokens: Any = ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce summary.json and annotation_units.csv for one completed case run."
    )
    parser.add_argument("--case", required=True, help="Case ID to evaluate, e.g., goofy.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing one completed pipeline run, e.g., case_outputs/goofy.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where summary.json and annotation_units.csv will be written.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run ID override. If omitted, the script tries config.json and then uses the case ID.",
    )
    parser.add_argument(
        "--case-name",
        default=None,
        help="Optional human-readable case name override.",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.85,
        help="Near-duplicate condition threshold using SequenceMatcher. Default: 0.85.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                raise ValueError(f"Expected JSON object in {path} at line {line_no}, got {type(obj).__name__}")
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: csv_cell(row.get(col, "")) for col in columns})


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def first_present(obj: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    """Return the first present/non-empty value from obj for any key in keys."""
    for key in keys:
        if key in obj and obj[key] not in (None, ""):
            return obj[key]
    return default


def ensure_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        # Accept the delimiter used by our own CSV and common comma-separated fields.
        if SAFE_DELIM in value:
            return [part.strip() for part in value.split(SAFE_DELIM) if part.strip()]
        if "," in value and not value.strip().startswith("["):
            return [part.strip() for part in value.split(",") if part.strip()]
        return [value.strip()] if value.strip() else []
    return [value]


def string_list(value: Any) -> list[str]:
    return [str(v) for v in ensure_list(value) if str(v).strip()]


def normalize_condition_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"pre", "precondition", "preconditions"}:
        return "precondition"
    if text in {"post", "postcondition", "postconditions"}:
        return "postcondition"
    return text


def normalize_text(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_count(text: Any) -> int:
    """Whitespace token count used by automatic length metrics."""
    return len(str(text or "").split())


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def mean(values: Iterable[Any]) -> float | None:
    nums = [float(v) for v in values if is_number(v)]
    if not nums:
        return None
    return sum(nums) / len(nums)


def is_number(value: Any) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(x)


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return SAFE_DELIM.join(csv_cell(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = str(item)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Artifact interpretation
# ---------------------------------------------------------------------------


def build_chunk_index(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = str(first_present(chunk, ["chunk_id", "id"], ""))
        if chunk_id:
            index[chunk_id] = chunk
    return index


def attack_step_id(step: dict[str, Any]) -> str:
    return str(first_present(step, ["attack_step_id", "step_id", "id"], ""))


def condition_id(condition: dict[str, Any], case_id: str, index: int) -> str:
    cid = first_present(condition, ["condition_id", "id", "field_id"], "")
    if cid:
        return str(cid)
    step_id = str(first_present(condition, ["attack_step_id", "step_id"], "unknown-step"))
    ctype = normalize_condition_type(first_present(condition, ["condition_type", "type"], "condition"))
    return f"{case_id}-{step_id}-{ctype}-{index:04d}"


def description_id(description: dict[str, Any], case_id: str) -> str:
    did = first_present(description, ["description_id", "generated_description_id", "field_id", "id"], "")
    if did:
        return str(did)
    step_id = str(first_present(description, ["attack_step_id", "step_id"], "unknown-step"))
    return f"{case_id}-{step_id}-desc01"


def description_text(description: dict[str, Any]) -> str:
    return str(first_present(description, ["description", "enriched_description", "generated_text", "text"], ""))


def condition_text(condition: dict[str, Any]) -> str:
    return str(first_present(condition, ["statement", "condition", "text", "generated_text"], ""))


def condition_evidence_ids(condition: dict[str, Any]) -> list[str]:
    return string_list(first_present(condition, ["evidence_chunk_ids", "input_chunk_ids", "chunk_ids"], []))


def description_evidence_ids(description: dict[str, Any]) -> list[str]:
    return string_list(first_present(description, ["evidence_chunk_ids", "input_chunk_ids", "chunk_ids"], []))


def get_step_metadata(step: dict[str, Any], case_id: str, case_name: str) -> dict[str, str]:
    metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
    raw_action = metadata.get("raw_action") if isinstance(metadata.get("raw_action"), dict) else {}

    step_id = attack_step_id(step)
    tactic_name = first_present(step, ["tactic_name", "attack_tactic", "tactic"], "")
    technique_id = first_present(step, ["technique_id", "attack_technique_id"], "")
    technique_name = first_present(step, ["technique_name"], "") or first_present(raw_action, ["name", "technique_name"], "")

    return {
        "case_id": case_id,
        "case_name": case_name,
        "milestone_id": str(first_present(step, ["milestone_id"], "")),
        "milestone_label": str(first_present(step, ["milestone_label", "milestone_name"], "")),
        "tactic_id": str(first_present(step, ["tactic_id"], "")),
        "tactic_name": str(tactic_name),
        "attack_step_id": step_id,
        "attack_step_label": str(first_present(step, ["attack_step_label", "original_label", "label", "name"], "")),
        "technique_id": str(technique_id),
        "technique_name": str(technique_name),
        "original_description": str(first_present(step, ["original_description", "input_text", "description"], "")),
    }


def infer_case_name(case_id: str, config: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    for key in ["case_name", "name", "title"]:
        if key in config and config[key]:
            return str(config[key])
    case_cfg = config.get("case") if isinstance(config.get("case"), dict) else {}
    for key in ["case_name", "name", "title"]:
        if key in case_cfg and case_cfg[key]:
            return str(case_cfg[key])
    return case_id.replace("_", " ").replace("-", " ").title()


def infer_run_id(case_id: str, config: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    for path in [
        ["run", "run_id"],
        ["run_id"],
        ["metadata", "run_id"],
    ]:
        value = nested_get(config, path)
        if value:
            return str(value)
    return case_id


def nested_get(obj: dict[str, Any], path: list[str], default: Any = "") -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def pipeline_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    models = config.get("models") if isinstance(config.get("models"), dict) else {}
    chunking = config.get("chunking") if isinstance(config.get("chunking"), dict) else {}
    generation = config.get("generation") if isinstance(config.get("generation"), dict) else {}

    return {
        "retrieval_strategy": retrieval.get("strategy", ""),
        "bm25_k": retrieval.get("bm25_k", ""),
        "vector_k": retrieval.get("vector_k", ""),
        "top_k": retrieval.get("top_k", ""),
        "hybrid_alpha": retrieval.get("hybrid_alpha", ""),
        "chunker": chunking.get("strategy", chunking.get("chunker", "")),
        "chunk_size": chunking.get("max_chunk_size", chunking.get("chunk_size", "")),
        "embedding_model": models.get("embeddings", models.get("sentence_transformer", "")),
        "slm_model": models.get("text_generation", models.get("slm_model", models.get("model", ""))),
        "prompt_template_id": generation.get("prompt_template_id", generation.get("template_id", "")),
        "tokenizer": "whitespace",
    }


def retrieval_config_values(config: dict[str, Any]) -> dict[str, Any]:
    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    return {
        "retrieval_strategy": retrieval.get("strategy", ""),
        "bm25_k": retrieval.get("bm25_k", ""),
        "vector_k": retrieval.get("vector_k", ""),
        "top_k": retrieval.get("top_k", ""),
    }


# ---------------------------------------------------------------------------
# Retrieval/generation metadata extraction
# ---------------------------------------------------------------------------


def build_retrieval_metadata(
    retrieval_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    chunk_index: dict[str, dict[str, Any]],
) -> dict[str, RetrievalMetadata]:
    """Build one retrieval metadata object per attack step.

    The production artifacts may encode retrieval outputs either as one row per
    attack step with a nested list of chunks, or as one row per retrieved chunk.
    This function accepts both shapes.
    """
    meta: dict[str, RetrievalMetadata] = defaultdict(RetrievalMetadata)

    for row in retrieval_rows:
        step_id = str(first_present(row, ["attack_step_id", "step_id"], ""))
        if not step_id:
            continue

        m = meta[step_id]
        if not m.query:
            m.query = str(first_present(row, ["query_text", "query", "retrieval_query"], ""))
        if not m.latency_ms:
            m.latency_ms = first_present(row, ["retrieval_latency_ms", "latency_ms"], "")

        # Nested retrieval result list.
        nested = first_present(row, ["retrieved_chunks", "results", "chunks", "documents"], [])
        if isinstance(nested, list) and nested:
            for idx, item in enumerate(nested, start=1):
                if isinstance(item, dict):
                    chunk_id = str(first_present(item, ["chunk_id", "id", "document_id"], ""))
                    rank = first_present(item, ["rank"], idx)
                else:
                    chunk_id = str(item)
                    rank = idx
                if chunk_id:
                    m.retrieved_chunk_ids.append(chunk_id)
                    if chunk_id not in m.chunk_rank:
                        m.chunk_rank[chunk_id] = int(rank) if str(rank).isdigit() else idx

        # Flat one-row-per-chunk shape.
        flat_chunk_id = str(first_present(row, ["chunk_id", "retrieved_chunk_id", "evidence_chunk_id"], ""))
        if flat_chunk_id:
            rank = first_present(row, ["rank", "retrieval_rank"], len(m.retrieved_chunk_ids) + 1)
            m.retrieved_chunk_ids.append(flat_chunk_id)
            if flat_chunk_id not in m.chunk_rank:
                m.chunk_rank[flat_chunk_id] = int(rank) if str(rank).isdigit() else len(m.retrieved_chunk_ids)

    # Runtime rows can provide retrieval latency even when retrieval rows do not.
    for row in runtime_rows:
        step_id = str(first_present(row, ["attack_step_id", "step_id"], ""))
        if not step_id:
            continue
        event = str(first_present(row, ["event", "stage", "component"], "")).lower()
        if "retriev" not in event:
            continue
        m = meta[step_id]
        if not m.latency_ms:
            m.latency_ms = first_present(row, ["latency_ms", "duration_ms", "retrieval_latency_ms"], "")

    # Remove duplicates while preserving rank/order.
    for m in meta.values():
        m.retrieved_chunk_ids = dedupe_preserve_order(m.retrieved_chunk_ids)
        # If no explicit rank was present, assign list order.
        for idx, chunk_id in enumerate(m.retrieved_chunk_ids, start=1):
            m.chunk_rank.setdefault(chunk_id, idx)

    return dict(meta)


def build_generation_metadata(
    generation_logs: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, GenerationMetadata]:
    meta: dict[str, GenerationMetadata] = defaultdict(GenerationMetadata)
    default_model = pipeline_config_summary(config).get("slm_model", "")
    default_prompt = pipeline_config_summary(config).get("prompt_template_id", "")

    for row in generation_logs:
        step_id = str(first_present(row, ["attack_step_id", "step_id"], ""))
        if not step_id:
            continue
        m = meta[step_id]
        m.model = str(first_present(row, ["model_name", "model", "generation_model"], m.model or default_model))
        m.prompt_template_id = str(
            first_present(row, ["prompt_template_id", "template_id", "prompt_version"], m.prompt_template_id or default_prompt)
        )
        m.latency_ms = first_present(row, ["generation_latency_ms", "latency_ms", "duration_ms"], m.latency_ms)
        m.context_tokens = first_present(row, ["context_tokens", "input_tokens", "prompt_tokens"], m.context_tokens)
        m.output_tokens = first_present(row, ["output_tokens", "completion_tokens"], m.output_tokens)

    for row in runtime_rows:
        step_id = str(first_present(row, ["attack_step_id", "step_id"], ""))
        if not step_id:
            continue
        event = str(first_present(row, ["event", "stage", "component"], "")).lower()
        if "generation" not in event and "generate" not in event and "llm" not in event:
            continue
        m = meta[step_id]
        m.model = m.model or str(first_present(row, ["model_name", "model"], default_model))
        m.prompt_template_id = m.prompt_template_id or str(first_present(row, ["prompt_template_id", "template_id"], default_prompt))
        m.latency_ms = m.latency_ms or first_present(row, ["latency_ms", "duration_ms", "generation_latency_ms"], "")
        m.context_tokens = m.context_tokens or first_present(row, ["context_tokens", "input_tokens", "prompt_tokens"], "")
        m.output_tokens = m.output_tokens or first_present(row, ["output_tokens", "completion_tokens"], "")

    # Populate defaults for steps that only appear at annotation-row creation time.
    for m in meta.values():
        m.model = m.model or str(default_model)
        m.prompt_template_id = m.prompt_template_id or str(default_prompt)

    return dict(meta)


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


def compute_summary(
    *,
    case_id: str,
    case_name: str,
    run_id: str,
    config: dict[str, Any],
    attack_steps: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    descriptions: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    retrieval_meta: dict[str, RetrievalMetadata],
    generation_meta: dict[str, GenerationMetadata],
    chunk_index: dict[str, dict[str, Any]],
    duplicate_threshold: float,
) -> dict[str, Any]:
    step_ids = [attack_step_id(step) for step in attack_steps if attack_step_id(step)]
    step_id_set = set(step_ids)
    total_steps = len(step_ids)

    desc_by_step = {str(first_present(d, ["attack_step_id", "step_id"], "")): d for d in descriptions}
    conds_by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cond in conditions:
        conds_by_step[str(first_present(cond, ["attack_step_id", "step_id"], ""))].append(cond)

    preconditions = [c for c in conditions if normalize_condition_type(first_present(c, ["condition_type", "type"], "")) == "precondition"]
    postconditions = [c for c in conditions if normalize_condition_type(first_present(c, ["condition_type", "type"], "")) == "postcondition"]

    generated_description_fields = [d for d in descriptions if description_text(d).strip()]
    generated_pre_fields = [c for c in preconditions if condition_text(c).strip()]
    generated_post_fields = [c for c in postconditions if condition_text(c).strip()]

    milestones = set()
    for step in attack_steps:
        mid = first_present(step, ["milestone_id"], "")
        if mid:
            milestones.add(str(mid))

    documents = set()
    for chunk in chunks:
        did = first_present(chunk, ["document_id", "doc_id", "source_path", "title"], "")
        if did:
            documents.add(str(did))

    desc_step_count = sum(1 for sid in step_ids if description_text(desc_by_step.get(sid, {})).strip())
    pre_step_count = sum(
        1
        for sid in step_ids
        if any(
            normalize_condition_type(first_present(c, ["condition_type", "type"], "")) == "precondition"
            and condition_text(c).strip()
            for c in conds_by_step.get(sid, [])
        )
    )
    post_step_count = sum(
        1
        for sid in step_ids
        if any(
            normalize_condition_type(first_present(c, ["condition_type", "type"], "")) == "postcondition"
            and condition_text(c).strip()
            for c in conds_by_step.get(sid, [])
        )
    )
    both_step_count = sum(
        1
        for sid in step_ids
        if any(normalize_condition_type(first_present(c, ["condition_type", "type"], "")) == "precondition" for c in conds_by_step.get(sid, []))
        and any(normalize_condition_type(first_present(c, ["condition_type", "type"], "")) == "postcondition" for c in conds_by_step.get(sid, []))
    )

    notes: list[str] = []
    if not descriptions:
        notes.append("No generated description records found.")
    if not conditions:
        notes.append("No generated/final condition records found.")
    if not chunks:
        notes.append("No chunks.jsonl records found; evidence-link validation could not be performed.")

    duplicate_pre = duplicate_condition_rate(preconditions, duplicate_threshold)
    duplicate_post = duplicate_condition_rate(postconditions, duplicate_threshold)
    notes.append(
        "Duplicate-condition rate computed by normalized exact/near-duplicate fallback "
        f"within the same attack step, threshold={duplicate_threshold}."
    )

    retrieval_latencies = [m.latency_ms for sid, m in retrieval_meta.items() if sid in step_id_set and is_number(m.latency_ms)]
    generation_latencies = [m.latency_ms for sid, m in generation_meta.items() if sid in step_id_set and is_number(m.latency_ms)]
    context_tokens = [m.context_tokens for sid, m in generation_meta.items() if sid in step_id_set and is_number(m.context_tokens)]
    retrieved_chunk_counts = [len(m.retrieved_chunk_ids or []) for sid, m in retrieval_meta.items() if sid in step_id_set]

    summary = {
        "case_id": case_id,
        "case_name": case_name,
        "run_id": run_id,
        "pipeline_config": pipeline_config_summary(config),
        "counts": {
            "documents": len(documents),
            "chunks": len(chunks),
            "milestones": len(milestones),
            "attack_steps": total_steps,
            "generated_descriptions": len(generated_description_fields),
            "generated_preconditions": len(generated_pre_fields),
            "generated_postconditions": len(generated_post_fields),
            "generated_conditions_total": len(generated_pre_fields) + len(generated_post_fields),
        },
        "automatic_metrics": {
            "field_coverage": {
                "descriptions": safe_div(desc_step_count, total_steps),
                "preconditions": safe_div(pre_step_count, total_steps),
                "postconditions": safe_div(post_step_count, total_steps),
            },
            "condition_coverage": {
                "at_least_one_precondition": safe_div(pre_step_count, total_steps),
                "at_least_one_postcondition": safe_div(post_step_count, total_steps),
                "at_least_one_precondition_and_postcondition": safe_div(both_step_count, total_steps),
            },
            "evidence_link_coverage": {
                "descriptions": evidence_link_coverage(generated_description_fields, "description", chunk_index),
                "preconditions": evidence_link_coverage(generated_pre_fields, "condition", chunk_index),
                "postconditions": evidence_link_coverage(generated_post_fields, "condition", chunk_index),
            },
            "average_evidence_count": {
                "descriptions": average_evidence_count(generated_description_fields, "description", chunk_index),
                "preconditions": average_evidence_count(generated_pre_fields, "condition", chunk_index),
                "postconditions": average_evidence_count(generated_post_fields, "condition", chunk_index),
            },
            "average_field_length_tokens": {
                "descriptions": mean(token_count(description_text(d)) for d in generated_description_fields),
                "preconditions": mean(token_count(condition_text(c)) for c in generated_pre_fields),
                "postconditions": mean(token_count(condition_text(c)) for c in generated_post_fields),
            },
            "duplicate_condition_rate": {
                "preconditions": duplicate_pre,
                "postconditions": duplicate_post,
            },
            "latency_ms": {
                "average_retrieval_per_attack_step": mean(retrieval_latencies),
                "average_generation_per_attack_step": mean(generation_latencies),
                "total_retrieval": sum(float(x) for x in retrieval_latencies),
                "total_generation": sum(float(x) for x in generation_latencies),
            },
            "retrieval_transferability": {
                "average_retrieved_context_tokens": mean(context_tokens),
                "average_retrieved_chunks_per_step": mean(retrieved_chunk_counts),
            },
        },
        "metric_denominators": {
            "field_coverage_attack_steps": total_steps,
            "description_fields": len(generated_description_fields),
            "precondition_fields": len(generated_pre_fields),
            "postcondition_fields": len(generated_post_fields),
            "retrieval_latency_records": len(retrieval_latencies),
            "generation_latency_records": len(generation_latencies),
        },
        "notes": notes,
    }
    return summary


def evidence_ids_for_field(field: dict[str, Any], field_kind: str) -> list[str]:
    if field_kind == "description":
        return description_evidence_ids(field)
    return condition_evidence_ids(field)


def valid_evidence_ids(field: dict[str, Any], field_kind: str, chunk_index: dict[str, dict[str, Any]]) -> list[str]:
    return [cid for cid in evidence_ids_for_field(field, field_kind) if cid in chunk_index]


def evidence_link_coverage(fields: list[dict[str, Any]], field_kind: str, chunk_index: dict[str, dict[str, Any]]) -> float | None:
    if not fields:
        return None
    linked = sum(1 for field in fields if valid_evidence_ids(field, field_kind, chunk_index))
    return linked / len(fields)


def average_evidence_count(fields: list[dict[str, Any]], field_kind: str, chunk_index: dict[str, dict[str, Any]]) -> float | None:
    if not fields:
        return None
    total = sum(len(valid_evidence_ids(field, field_kind, chunk_index)) for field in fields)
    return total / len(fields)


def duplicate_condition_rate(conditions: list[dict[str, Any]], threshold: float) -> float | None:
    """Estimate duplicate-condition rate within attack step.

    This fallback is used because the simplified pipeline has no canonicalization
    log. A condition is counted as duplicate if it is an exact normalized match
    or a near-duplicate of an earlier condition in the same attack step.
    """
    if not conditions:
        return None

    by_step: dict[str, list[str]] = defaultdict(list)
    for condition in conditions:
        step_id = str(first_present(condition, ["attack_step_id", "step_id"], ""))
        text = normalize_text(condition_text(condition))
        if text:
            by_step[step_id].append(text)

    total = 0
    duplicates = 0
    for texts in by_step.values():
        previous: list[str] = []
        for text in texts:
            total += 1
            if any(text == old or SequenceMatcher(None, text, old).ratio() >= threshold for old in previous):
                duplicates += 1
            else:
                previous.append(text)

    return safe_div(duplicates, total)


# ---------------------------------------------------------------------------
# Annotation CSV construction
# ---------------------------------------------------------------------------


def build_annotation_rows(
    *,
    case_id: str,
    case_name: str,
    run_id: str,
    config: dict[str, Any],
    attack_steps: list[dict[str, Any]],
    descriptions: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    retrieval_meta: dict[str, RetrievalMetadata],
    generation_meta: dict[str, GenerationMetadata],
    chunk_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    retrieval_cfg = retrieval_config_values(config)

    desc_by_step = {str(first_present(d, ["attack_step_id", "step_id"], "")): d for d in descriptions}
    conds_by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, cond in enumerate(conditions, start=1):
        # Ensure deterministic IDs if the generator did not already emit one.
        cond = dict(cond)
        cond.setdefault("condition_id", condition_id(cond, case_id, idx))
        conds_by_step[str(first_present(cond, ["attack_step_id", "step_id"], ""))].append(cond)

    for step in attack_steps:
        step_id = attack_step_id(step)
        if not step_id:
            continue

        step_meta = get_step_metadata(step, case_id, case_name)
        retrieval = retrieval_meta.get(step_id, RetrievalMetadata())
        generation = generation_meta.get(step_id, GenerationMetadata())

        description = desc_by_step.get(step_id)
        if description and description_text(description).strip():
            rows.append(
                build_description_row(
                    step_meta=step_meta,
                    run_id=run_id,
                    description=description,
                    retrieval=retrieval,
                    generation=generation,
                    retrieval_cfg=retrieval_cfg,
                    chunk_index=chunk_index,
                )
            )

        enriched_description = description_text(description or {})
        for condition in conds_by_step.get(step_id, []):
            if not condition_text(condition).strip():
                continue
            rows.append(
                build_condition_row(
                    step_meta=step_meta,
                    run_id=run_id,
                    condition=condition,
                    enriched_description=enriched_description,
                    retrieval=retrieval,
                    generation=generation,
                    retrieval_cfg=retrieval_cfg,
                    chunk_index=chunk_index,
                )
            )

    validate_annotation_rows(rows)
    return rows


def build_description_row(
    *,
    step_meta: dict[str, str],
    run_id: str,
    description: dict[str, Any],
    retrieval: RetrievalMetadata,
    generation: GenerationMetadata,
    retrieval_cfg: dict[str, Any],
    chunk_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    step_id = step_meta["attack_step_id"]
    field_id = description_id(description, step_meta["case_id"])
    text = description_text(description)
    evidence_ids = description_evidence_ids(description) or retrieval.retrieved_chunk_ids

    row = base_annotation_row(
        step_meta=step_meta,
        run_id=run_id,
        field_type="description",
        field_id=field_id,
        condition_type="",
        enriched_description=text,
        generated_field_text=text,
        evidence_ids=evidence_ids,
        retrieval=retrieval,
        generation=generation,
        retrieval_cfg=retrieval_cfg,
        chunk_index=chunk_index,
    )
    row["annotation_unit_id"] = f"{step_meta['case_id']}::{step_id}::description::{field_id}"

    # Condition criteria are not applicable to description rows.
    for col in CONDITION_MANUAL_COLUMNS:
        row[col] = NA
    for col in DESCRIPTION_MANUAL_COLUMNS:
        row[col] = ""
    return row


def build_condition_row(
    *,
    step_meta: dict[str, str],
    run_id: str,
    condition: dict[str, Any],
    enriched_description: str,
    retrieval: RetrievalMetadata,
    generation: GenerationMetadata,
    retrieval_cfg: dict[str, Any],
    chunk_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    step_id = step_meta["attack_step_id"]
    field_id = str(first_present(condition, ["condition_id", "id", "field_id"], ""))
    ctype = normalize_condition_type(first_present(condition, ["condition_type", "type"], ""))
    text = condition_text(condition)
    evidence_ids = condition_evidence_ids(condition) or retrieval.retrieved_chunk_ids

    row = base_annotation_row(
        step_meta=step_meta,
        run_id=run_id,
        field_type="condition",
        field_id=field_id,
        condition_type=ctype,
        enriched_description=enriched_description,
        generated_field_text=text,
        evidence_ids=evidence_ids,
        retrieval=retrieval,
        generation=generation,
        retrieval_cfg=retrieval_cfg,
        chunk_index=chunk_index,
    )
    row["annotation_unit_id"] = f"{step_meta['case_id']}::{step_id}::condition::{field_id}"

    # Description criteria are not applicable to condition rows.
    for col in CONDITION_MANUAL_COLUMNS:
        row[col] = ""
    for col in DESCRIPTION_MANUAL_COLUMNS:
        row[col] = NA
    return row


def base_annotation_row(
    *,
    step_meta: dict[str, str],
    run_id: str,
    field_type: str,
    field_id: str,
    condition_type: str,
    enriched_description: str,
    generated_field_text: str,
    evidence_ids: list[str],
    retrieval: RetrievalMetadata,
    generation: GenerationMetadata,
    retrieval_cfg: dict[str, Any],
    chunk_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence = evidence_payload(evidence_ids, chunk_index, retrieval.chunk_rank)

    row = {
        "annotation_unit_id": "",  # filled by caller
        "run_id": run_id,
        **step_meta,
        "field_type": field_type,
        "field_id": field_id,
        "condition_type": condition_type,
        "enriched_description": enriched_description,
        "generated_field_text": generated_field_text,
        **evidence,
        **retrieval_cfg,
        "retrieval_query": retrieval.query,
        "retrieval_latency_ms": retrieval.latency_ms,
        "generation_model": generation.model,
        "prompt_template_id": generation.prompt_template_id,
        "generation_latency_ms": generation.latency_ms,
        "context_tokens": generation.context_tokens,
        "output_tokens": generation.output_tokens,
    }
    return row


def evidence_payload(
    evidence_ids: list[str],
    chunk_index: dict[str, dict[str, Any]],
    rank_map: dict[str, int],
) -> dict[str, Any]:
    ids = dedupe_preserve_order(evidence_ids)
    chunks = [chunk_index[cid] for cid in ids if cid in chunk_index]

    # Keep only valid chunk IDs in evidence_chunk_ids. Invalid IDs are excluded
    # because annotators need exact evidence text for every cited chunk.
    valid_ids = [cid for cid in ids if cid in chunk_index]

    return {
        "evidence_chunk_ids": valid_ids,
        "evidence_document_ids": [first_present(ch, ["document_id", "doc_id"], "") for ch in chunks],
        "evidence_sources": [first_present(ch, ["source", "source_path"], "") for ch in chunks],
        "evidence_titles": [first_present(ch, ["title", "document_title"], "") for ch in chunks],
        "evidence_pages": [first_present(ch, ["page", "pages", "page_number"], "") for ch in chunks],
        "evidence_sections": [first_present(ch, ["section", "section_title", "heading"], "") for ch in chunks],
        "evidence_ranks": [rank_map.get(cid, "") for cid in valid_ids],
        "evidence_texts": [first_present(ch, ["text", "page_content", "content"], "") for ch in chunks],
    }


def validate_annotation_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        auid = str(row.get("annotation_unit_id", ""))
        if not auid:
            raise ValueError("Annotation row without annotation_unit_id")
        if auid in seen:
            duplicates.add(auid)
        seen.add(auid)

        if not row.get("case_id"):
            raise ValueError(f"Annotation row {auid} has empty case_id")
        if row.get("field_type") == "condition" and not row.get("condition_type"):
            raise ValueError(f"Condition row {auid} has empty condition_type")
        if row.get("field_type") == "description" and str(row.get("condition_type", "")) not in {"", NA}:
            raise ValueError(f"Description row {auid} should not have condition_type")
        if not str(row.get("generated_field_text", "")).strip():
            raise ValueError(f"Annotation row {auid} has empty generated_field_text")

        evidence_ids = string_list(row.get("evidence_chunk_ids", []))
        evidence_texts = string_list(row.get("evidence_texts", []))
        if len(evidence_ids) != len(evidence_texts):
            raise ValueError(
                f"Annotation row {auid} has {len(evidence_ids)} evidence IDs but "
                f"{len(evidence_texts)} evidence texts"
            )

    if duplicates:
        raise ValueError(f"Duplicate annotation_unit_id values: {sorted(duplicates)[:10]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    case_id = args.case
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    config = read_json(input_dir / "config.json", default={})
    case_name = infer_case_name(case_id, config, args.case_name)
    run_id = infer_run_id(case_id, config, args.run_id)

    # Core production artifacts.
    attack_steps = read_jsonl(input_dir / "attack_steps.jsonl")
    chunks = read_jsonl(input_dir / "chunks.jsonl")
    retrieval_rows = read_jsonl(input_dir / "retrieval_results.jsonl")
    runtime_rows = read_jsonl(input_dir / "runtime_metrics.jsonl")
    generation_logs = read_jsonl(input_dir / "generation_logs.jsonl")
    descriptions = read_jsonl(input_dir / "generated_descriptions.jsonl")

    # Prefer final conditions. In the no-verifier pipeline this is normally a
    # copy of generated_conditions.jsonl, but using it keeps the evaluator stable
    # if later post-processing is reintroduced.
    conditions_path = input_dir / "final_conditions.jsonl"
    if not conditions_path.exists():
        conditions_path = input_dir / "generated_conditions.jsonl"
    conditions = read_jsonl(conditions_path)

    if not attack_steps:
        raise ValueError(f"No attack steps found in {input_dir / 'attack_steps.jsonl'}")

    chunk_index = build_chunk_index(chunks)
    retrieval_meta = build_retrieval_metadata(retrieval_rows, runtime_rows, chunk_index)
    generation_meta = build_generation_metadata(generation_logs, runtime_rows, config)

    summary = compute_summary(
        case_id=case_id,
        case_name=case_name,
        run_id=run_id,
        config=config,
        attack_steps=attack_steps,
        chunks=chunks,
        descriptions=descriptions,
        conditions=conditions,
        retrieval_meta=retrieval_meta,
        generation_meta=generation_meta,
        chunk_index=chunk_index,
        duplicate_threshold=args.duplicate_threshold,
    )

    annotation_rows = build_annotation_rows(
        case_id=case_id,
        case_name=case_name,
        run_id=run_id,
        config=config,
        attack_steps=attack_steps,
        descriptions=descriptions,
        conditions=conditions,
        retrieval_meta=retrieval_meta,
        generation_meta=generation_meta,
        chunk_index=chunk_index,
    )

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "annotation_units.csv", annotation_rows, ANNOTATION_COLUMNS)

    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'annotation_units.csv'}")
    print(f"Annotation rows: {len(annotation_rows)}")


if __name__ == "__main__":
    main()
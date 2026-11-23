from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from tqdm import tqdm

# Scripts live in evaluation/rag/squad. Add repository root so production
# modules are imported exactly as they are in the pipeline.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retrieval import build_retriever, retrieve_with_provenance

try:
    from ragas import SingleTurnSample
except Exception:  # depends on the installed Ragas version
    from ragas.dataset_schema import SingleTurnSample

from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall


DEFAULT_COMPONENT_K_VALUES = [1, 3, 5, 10, 20, 50, 100, 200]
DEFAULT_HYBRID_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass
class ContextRecord:
    context_id: str
    title: str
    text: str
    source_index: int


@dataclass
class QueryRecord:
    query_id: str
    context_id: str
    title: str
    question: str
    answer_text: str
    answer_start: int
    answer_end: int


@dataclass
class ChunkRecord:
    chunk_id: str
    context_id: str
    title: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    token_count: int


@dataclass
class QrelRecord:
    query_id: str
    chunk_id: str
    context_id: str
    relevance: int
    answer_text: str
    answer_start: int
    answer_end: int
    match_type: str


@dataclass
class RetrievedItem:
    rank: int
    chunk_id: str
    context_id: str
    text: str
    token_count: int


class RagasEvaluator:
    """Small wrapper around Ragas non-LLM context metrics."""

    def __init__(self) -> None:
        self.precision_metric = NonLLMContextPrecisionWithReference()
        self.recall_metric = NonLLMContextRecall()

    def score(self, retrieved_contexts: list[str], reference_contexts: list[str]) -> tuple[float, float, float, float]:
        if not reference_contexts:
            return 0.0, 0.0, 0.0, 0.0
        sample = SingleTurnSample(
            retrieved_contexts=retrieved_contexts,
            reference_contexts=reference_contexts,
        )
        precision = to_float(self.precision_metric.single_turn_score(sample))
        recall = to_float(self.recall_metric.single_turn_score(sample))
        f1 = harmonic(precision, recall, beta=1.0)
        f2 = harmonic(precision, recall, beta=2.0)
        return precision, recall, f1, f2


# ---------------------------------------------------------------------------
# Loading files produced by chunks.py
# ---------------------------------------------------------------------------


def load_prepared_inputs(input_dir: Path) -> tuple[list[ChunkRecord], list[QueryRecord], dict[str, set[str]]]:
    chunks = read_chunks(input_dir / "chunks.jsonl")
    queries = read_queries(input_dir / "queries.csv")
    qrels = read_qrels(input_dir / "qrels.csv")
    qrels_by_query = group_qrels(qrels)
    # Keep only queries with at least one answer-bearing chunk. This should
    # already be true if chunks.py completed successfully, but this guard keeps
    # retriever scripts robust to partially edited inputs.
    queries = [q for q in queries if q.query_id in qrels_by_query]
    return chunks, queries, qrels_by_query


def read_chunks(path: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            chunks.append(
                ChunkRecord(
                    chunk_id=str(item["chunk_id"]),
                    context_id=str(item["context_id"]),
                    title=str(item.get("title", "")),
                    chunk_index=int(item.get("chunk_index", 0)),
                    text=str(item["text"]),
                    start_char=int(item.get("start_char", -1)),
                    end_char=int(item.get("end_char", -1)),
                    token_count=int(item.get("token_count", count_tokens(str(item["text"])))),
                )
            )
    return chunks


def read_queries(path: Path) -> list[QueryRecord]:
    rows = read_csv_dicts(path)
    return [
        QueryRecord(
            query_id=str(row["query_id"]),
            context_id=str(row["context_id"]),
            title=str(row.get("title", "")),
            question=str(row["question"]),
            answer_text=str(row["answer_text"]),
            answer_start=int(row["answer_start"]),
            answer_end=int(row["answer_end"]),
        )
        for row in rows
    ]


def read_qrels(path: Path) -> list[QrelRecord]:
    rows = read_csv_dicts(path)
    return [
        QrelRecord(
            query_id=str(row["query_id"]),
            chunk_id=str(row["chunk_id"]),
            context_id=str(row["context_id"]),
            relevance=int(row.get("relevance", 1)),
            answer_text=str(row.get("answer_text", "")),
            answer_start=int(row.get("answer_start", -1)),
            answer_end=int(row.get("answer_end", -1)),
            match_type=str(row.get("match_type", "")),
        )
        for row in rows
    ]


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_qrels(qrels: list[QrelRecord]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for qrel in qrels:
        grouped.setdefault(qrel.query_id, set()).add(qrel.chunk_id)
    return grouped


# ---------------------------------------------------------------------------
# Production retriever setup
# ---------------------------------------------------------------------------


def chunks_to_documents(chunks: list[ChunkRecord]) -> list[Document]:
    """Convert saved chunks into LangChain Documents for build_retriever()."""
    return [
        Document(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.context_id,
                "context_id": chunk.context_id,
                "source": f"squad::{chunk.context_id}",
                "source_path": f"squad::{chunk.context_id}",
                "title": chunk.title,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
            },
        )
        for chunk in chunks
    ]


def retrieval_config(
    *,
    strategy: str,
    top_k: int,
    embedding_backend: str,
    embedding_model: str,
    bm25_k: int | None = None,
    vector_k: int | None = None,
    bm25_weight: float | None = None,
) -> dict[str, Any]:
    retrieval: dict[str, Any] = {"enabled": True, "strategy": strategy, "top_k": int(top_k)}
    if strategy == "hybrid":
        retrieval.update(
            {
                "bm25_k": int(bm25_k or top_k),
                "vector_k": int(vector_k or top_k),
                "hybrid_alpha": 0.5 if bm25_weight is None else float(bm25_weight),
            }
        )
    return {
        "retrieval": retrieval,
        "models": {
            "embedding_backend": embedding_backend,
            "embeddings": embedding_model,
            "sentence_transformer": embedding_model,
        },
    }


def normalize_retrieval_results(record: Any, chunk_lookup: dict[str, ChunkRecord], *, max_k: int) -> list[RetrievedItem]:
    """Convert pipeline provenance records into a compact ranked list."""
    output: list[RetrievedItem] = []
    seen: set[str] = set()
    for result in getattr(record, "results", []):
        chunk_id = str(getattr(result, "chunk_id", "") or "")
        text = str(getattr(result, "text", "") or "")
        if not chunk_id:
            metadata = getattr(result, "metadata", {}) or {}
            chunk_id = str(metadata.get("chunk_id", ""))
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = chunk_lookup.get(chunk_id)
        if chunk:
            context_id = chunk.context_id
            text = text or chunk.text
            token_count = chunk.token_count
        else:
            context_id = str(getattr(result, "document_id", ""))
            token_count = count_tokens(text)
        output.append(
            RetrievedItem(
                rank=len(output) + 1,
                chunk_id=chunk_id,
                context_id=context_id,
                text=text,
                token_count=token_count,
            )
        )
        if len(output) >= max_k:
            break
    return output


# ---------------------------------------------------------------------------
# Shared evaluation logic
# ---------------------------------------------------------------------------


def evaluate_prefix_retrieval(
    *,
    retriever_name: str,
    run_id: str,
    documents: list[Document],
    chunks: list[ChunkRecord],
    queries: list[QueryRecord],
    qrels_by_query: dict[str, set[str]],
    k_values: list[int],
    max_k: int,
    config: dict[str, Any],
    out_dir: Path,
    extra_columns: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Retrieve once at max_k and evaluate all requested top-k prefixes.

    This is intentionally fast: the retriever returns the maximum required
    candidate set once per query, and the script computes Ragas scores for
    prefixes such as k=1,3,5,... from the same ranking.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit").mkdir(parents=True, exist_ok=True)

    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    ragas = RagasEvaluator()
    extra_columns = extra_columns or {}

    build_start = time.perf_counter()
    bundle = build_retriever(documents, config)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    per_query_rows: list[dict[str, Any]] = []
    audit_path = out_dir / "audit" / f"{run_id}_top{max_k}.jsonl"

    with audit_path.open("w", encoding="utf-8") as audit_f:
        for query in tqdm(queries, desc=f"Retrieving {run_id}@{max_k}"):
            relevant_ids = qrels_by_query.get(query.query_id, set())
            if not relevant_ids:
                continue
            reference_contexts = [chunk_lookup[cid].text for cid in sorted(relevant_ids) if cid in chunk_lookup]

            query_start = time.perf_counter()
            record = retrieve_with_provenance(
                bundle,
                query=query.question,
                run_id=run_id,
                case_id="squad",
                attack_step_id=query.query_id,
            )
            latency_ms = (time.perf_counter() - query_start) * 1000.0
            retrieved = normalize_retrieval_results(record, chunk_lookup, max_k=max_k)

            audit_f.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "retriever": retriever_name,
                        "query_id": query.query_id,
                        "question": query.question,
                        "answer_text": query.answer_text,
                        "gold_chunk_ids": sorted(relevant_ids),
                        "max_k_retrieved": max_k,
                        "retrieved": [asdict(item) for item in retrieved],
                        **extra_columns,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            for k in k_values:
                top = retrieved[:k]
                retrieved_ids = [item.chunk_id for item in top]
                retrieved_texts = [item.text for item in top]
                ragas_p, ragas_r, ragas_f1, ragas_f2 = ragas.score(retrieved_texts, reference_contexts)
                rank_metrics = ranking_metrics(retrieved_ids, relevant_ids, k)

                per_query_rows.append(
                    {
                        "run_id": run_id,
                        "retriever": retriever_name,
                        "k": int(k),
                        "query_id": query.query_id,
                        "context_id": query.context_id,
                        "ragas_precision": ragas_p,
                        "ragas_recall": ragas_r,
                        "ragas_f1": ragas_f1,
                        "ragas_f2": ragas_f2,
                        "hit": rank_metrics["hit"],
                        "mrr": rank_metrics["mrr"],
                        "ndcg": rank_metrics["ndcg"],
                        "context_tokens": sum(item.token_count for item in top),
                        "latency_ms": latency_ms,
                        "max_k_retrieved": int(max_k),
                        "num_relevant_chunks": len(relevant_ids),
                        "num_retrieved_chunks": len(top),
                        **extra_columns,
                    }
                )

    aggregate_rows = aggregate_metrics(per_query_rows, build_ms=build_ms)
    return per_query_rows, aggregate_rows, build_ms


def aggregate_metrics(per_query_rows: list[dict[str, Any]], *, build_ms: float) -> list[dict[str, Any]]:
    """Aggregate the metrics requested for paper-level CSV tables."""
    if not per_query_rows:
        return []
    df = pd.DataFrame(per_query_rows)
    group_cols = [
        col
        for col in ["run_id", "retriever", "bm25_k", "vector_k", "bm25_weight", "vector_weight", "k", "max_k_retrieved"]
        if col in df.columns
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: none_if_nan(value) for col, value in zip(group_cols, keys)}
        row.update(
            {
                "ragas_precision": float(group["ragas_precision"].mean()),
                "ragas_recall": float(group["ragas_recall"].mean()),
                "ragas_f1": float(group["ragas_f1"].mean()),
                "ragas_f2": float(group["ragas_f2"].mean()),
                "hit": float(group["hit"].mean()),
                "mrr": float(group["mrr"].mean()),
                "ndcg": float(group["ndcg"].mean()),
                "context_tokens_mean": float(group["context_tokens"].mean()),
                "context_tokens_median": float(group["context_tokens"].median()),
                "context_tokens_p95": percentile(group["context_tokens"], 95),
                "latency_ms_mean": float(group["latency_ms"].mean()),
                "latency_ms_median": float(group["latency_ms"].median()),
                "latency_ms_p95": percentile(group["latency_ms"], 95),
                "retriever_build_ms": float(build_ms),
                "num_queries": int(len(group)),
            }
        )
        rows.append(row)
    return rows


def ranking_metrics(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> dict[str, float]:
    top_ids = retrieved_ids[:k]
    rel_count = len(set(top_ids) & relevant_ids)
    return {
        "hit": 1.0 if rel_count > 0 else 0.0,
        "mrr": reciprocal_rank(top_ids, relevant_ids),
        "ndcg": ndcg(top_ids, relevant_ids, k),
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Preserve column order from the first row and append any later columns.
    fieldnames: list[str] = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def harmonic(precision: float, recall: float, *, beta: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for idx, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / idx
    return 0.0


def ndcg(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    gains = [1.0 if cid in relevant_ids else 0.0 for cid in retrieved_ids[:k]]
    dcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(gains))
    ideal_len = min(len(relevant_ids), k)
    if ideal_len == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_len))
    return dcg / idcg if idcg else 0.0


def percentile(values: Iterable[Any], p: float) -> float:
    values = [float(v) for v in values]
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def count_tokens(text: str) -> int:
    return len(str(text).split())


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def none_if_nan(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except Exception:
        pass
    return value


def format_weight(weight: float) -> str:
    return str(float(weight)).replace(".", "p")

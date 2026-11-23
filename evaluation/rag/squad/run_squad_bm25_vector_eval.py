#!/usr/bin/env python3
"""
Fast BM25/vector retrieval evaluation on SQuAD using Ragas metrics.

Speed policy
------------
For each retriever, the script retrieves once at max(k_values) for each query,
then evaluates the prefixes k = 1, 3, 5, ... from the same ranked list.
This avoids rebuilding and querying the same retriever for every k.

Consequence: latency_ms_* is the latency for retrieving max(k_values), repeated
on every k row. Context-token metrics and effectiveness metrics are computed on
true top-k prefixes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from chromadb.utils import embedding_functions
from chunking_evaluation.chunking import ClusterSemanticChunker
from datasets import Dataset, load_dataset
from langchain_core.documents import Document
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retrieval import build_retriever, retrieve_with_provenance

try:
    from ragas import SingleTurnSample
except Exception:
    from ragas.dataset_schema import SingleTurnSample

from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall


DEFAULT_K_VALUES = [1, 3, 5, 10, 20, 50, 100, 200]
METRIC_COLUMNS = [
    "ragas_precision",
    "ragas_recall",
    "ragas_f1",
    "ragas_f2",
    "hit",
    "mrr",
    "ndcg",
    "context_tokens_mean",
    "context_tokens_median",
    "context_tokens_p95",
    "latency_ms_mean",
    "latency_ms_median",
    "latency_ms_p95",
    "retriever_build_ms",
    "num_queries",
]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast BM25/vector SQuAD retrieval evaluation.")
    parser.add_argument("--dataset", default="rajpurkar/squad")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="evaluation/rag/results/squad_components_fast")
    parser.add_argument("--limit-contexts", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--retrievers", nargs="+", choices=["bm25", "vector"], default=["bm25", "vector"])
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--embedding-backend", default="sentence_transformer")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--fail-on-empty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit").mkdir(parents=True, exist_ok=True)

    k_values = sorted({int(k) for k in args.k_values if int(k) > 0})
    if not k_values:
        raise ValueError("At least one positive k value is required")
    max_k = max(k_values)

    print(f"Loading {args.dataset!r}, split={args.split!r}")
    ds = load_dataset(args.dataset, split=args.split)

    # Keep all QA pairs for each selected context. This preserves the SQuAD
    # many-questions-per-context structure.
    contexts, queries = build_contexts_and_queries(
        ds,
        limit_contexts=args.limit_contexts,
        limit_questions=args.limit_questions,
    )

    print("Chunking contexts")
    chunks = chunk_contexts(contexts, max_chunk_size=args.chunk_size)

    # Build answer-span grounded relevance labels. Only answer-bearing chunks
    # are reference contexts for Ragas.
    qrels = build_qrels(queries, chunks)
    qrels_by_query = group_qrels(qrels)
    queries = [q for q in queries if q.query_id in qrels_by_query]

    if args.fail_on_empty:
        if not contexts:
            raise ValueError("No contexts loaded")
        if not chunks:
            raise ValueError("No chunks generated")
        if not queries:
            raise ValueError("No usable queries after qrel filtering")
        if not qrels:
            raise ValueError("No qrels generated")

    print(f"Contexts: {len(contexts)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Usable queries: {len(queries)}")
    print(f"Qrels: {len(qrels)}")
    print(f"Fast retrieval mode: retrieve top-{max_k} once, evaluate prefixes {k_values}")

    # Persist the prepared corpus so the hybrid script can reuse it without
    # rechunking SQuAD.
    write_inputs(out_dir, contexts, queries, chunks, qrels)

    documents = chunks_to_documents(chunks)
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    ragas_runner = RagasRunner()

    all_aggregate: list[dict[str, Any]] = []
    all_per_query: list[dict[str, Any]] = []

    for retriever in args.retrievers:
        print(f"Running {retriever}: retrieving top-{max_k} once per query")
        per_query, aggregate = evaluate_retriever_prefixes(
            retriever=retriever,
            k_values=k_values,
            max_k=max_k,
            documents=documents,
            queries=queries,
            chunk_lookup=chunk_lookup,
            qrels_by_query=qrels_by_query,
            ragas_runner=ragas_runner,
            out_dir=out_dir,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
        )
        all_per_query.extend(per_query)
        all_aggregate.extend(aggregate)
        write_csv(out_dir / f"{retriever}_by_k.csv", aggregate)

    write_csv(out_dir / "component_by_k.csv", all_aggregate)
    write_csv(out_dir / "component_per_query.csv", all_per_query)
    write_json(
        out_dir / "run_report.json",
        {
            "dataset": args.dataset,
            "split": args.split,
            "k_values": k_values,
            "max_k_retrieved": max_k,
            "retrievers": args.retrievers,
            "num_contexts": len(contexts),
            "num_chunks": len(chunks),
            "num_queries": len(queries),
            "num_qrels": len(qrels),
            "metrics": METRIC_COLUMNS,
            "latency_policy": (
                "retrieval is executed once per query at max(k_values); "
                "latency_ms_* is top-max_k latency, while metrics are evaluated on top-k prefixes"
            ),
        },
    )
    print(f"Done. CSV files written to {out_dir}")


def evaluate_retriever_prefixes(
    *,
    retriever: str,
    k_values: list[int],
    max_k: int,
    documents: list[Document],
    queries: list[QueryRecord],
    chunk_lookup: dict[str, ChunkRecord],
    qrels_by_query: dict[str, set[str]],
    ragas_runner: "RagasRunner",
    out_dir: Path,
    embedding_backend: str,
    embedding_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_id = f"{retriever}_top{max_k}_prefix"
    config = retrieval_config(
        strategy=retriever,
        top_k=max_k,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
    )

    build_start = time.perf_counter()
    bundle = build_retriever(documents, config)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    per_query_rows: list[dict[str, Any]] = []
    audit_path = out_dir / "audit" / f"{run_id}_retrieved_chunks.jsonl"

    with audit_path.open("w", encoding="utf-8") as audit_f:
        for query in tqdm(queries, desc=f"Retrieving {run_id}"):
            relevant_ids = qrels_by_query.get(query.query_id, set())
            if not relevant_ids:
                continue
            reference_contexts = [chunk_lookup[cid].text for cid in sorted(relevant_ids) if cid in chunk_lookup]

            start = time.perf_counter()
            record = retrieve_with_provenance(bundle, query=query.question, run_id=run_id, case_id="squad", attack_step_id=query.query_id)
            latency_ms = (time.perf_counter() - start) * 1000.0

            retrieved = normalize_retrieval_results(record, chunk_lookup, max_k=max_k)
            audit_f.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "retriever": retriever,
                        "max_k_retrieved": max_k,
                        "query_id": query.query_id,
                        "question": query.question,
                        "answer_text": query.answer_text,
                        "gold_chunk_ids": sorted(relevant_ids),
                        "retrieved": [asdict(item) for item in retrieved],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            for k in k_values:
                top = retrieved[:k]
                retrieved_ids = [item.chunk_id for item in top]
                retrieved_texts = [item.text for item in top]
                ragas_precision, ragas_recall = ragas_runner.score(retrieved_texts, reference_contexts)
                ranking = ranking_metrics(retrieved_ids, relevant_ids, k)
                per_query_rows.append(
                    {
                        "run_id": run_id,
                        "retriever": retriever,
                        "k": k,
                        "max_k_retrieved": max_k,
                        "query_id": query.query_id,
                        "ragas_precision": ragas_precision,
                        "ragas_recall": ragas_recall,
                        "ragas_f1": harmonic(ragas_precision, ragas_recall, beta=1.0),
                        "ragas_f2": harmonic(ragas_precision, ragas_recall, beta=2.0),
                        "hit": ranking["hit"],
                        "mrr": ranking["mrr"],
                        "ndcg": ranking["ndcg"],
                        "context_tokens": sum(item.token_count for item in top),
                        "latency_ms": latency_ms,
                    }
                )

    aggregate_rows = []
    for k, group in pd.DataFrame(per_query_rows).groupby("k", dropna=False):
        aggregate_rows.append(
            aggregate_metrics(
                group.to_dict(orient="records"),
                retriever_build_ms=build_ms,
                identifiers={"retriever": retriever, "k": int(k), "max_k_retrieved": max_k},
            )
        )
    return per_query_rows, sorted(aggregate_rows, key=lambda row: row["k"])


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def build_contexts_and_queries(
    ds: Dataset,
    *,
    limit_contexts: int | None,
    limit_questions: int | None,
) -> tuple[list[ContextRecord], list[QueryRecord]]:
    contexts_by_text: dict[str, ContextRecord] = {}
    queries: list[QueryRecord] = []

    for idx, item in enumerate(tqdm(ds, desc="Loading SQuAD rows")):
        context = str(item.get("context", ""))
        question = str(item.get("question", "")).strip()
        title = str(item.get("title", ""))
        answers = item.get("answers") or {}
        answer_texts = answers.get("text") or []
        answer_starts = answers.get("answer_start") or []
        if not context.strip() or not question or not answer_texts or not answer_starts:
            continue

        if context not in contexts_by_text:
            if limit_contexts is not None and len(contexts_by_text) >= limit_contexts:
                continue
            context_id = make_context_id(title=title, text=context, index=len(contexts_by_text))
            contexts_by_text[context] = ContextRecord(context_id=context_id, title=title, text=context, source_index=idx)

        ctx = contexts_by_text.get(context)
        if ctx is None:
            continue

        answer_text = str(answer_texts[0])
        answer_start = int(answer_starts[0])
        answer_end = answer_start + len(answer_text)
        if not answer_text:
            continue

        query_id = str(item.get("id") or f"q_{stable_hash(question + answer_text)[:12]}")
        queries.append(
            QueryRecord(
                query_id=query_id,
                context_id=ctx.context_id,
                title=title,
                question=question,
                answer_text=answer_text,
                answer_start=answer_start,
                answer_end=answer_end,
            )
        )
        if limit_questions is not None and len(queries) >= limit_questions:
            break

    return list(contexts_by_text.values()), queries


def chunk_contexts(contexts: list[ContextRecord], *, max_chunk_size: int) -> list[ChunkRecord]:
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    chunker = ClusterSemanticChunker(default_ef, max_chunk_size=max_chunk_size)
    chunks: list[ChunkRecord] = []

    for ctx in tqdm(contexts, desc="Chunking contexts"):
        cursor = 0
        for chunk_index, raw_chunk in enumerate(chunker.split_text(ctx.text), start=1):
            text = str(raw_chunk).strip()
            if not text:
                continue
            start_char, end_char = locate_chunk(ctx.text, text, cursor)
            if start_char >= 0:
                cursor = max(cursor, end_char)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{ctx.context_id}_c{chunk_index:05d}",
                    context_id=ctx.context_id,
                    title=ctx.title,
                    chunk_index=chunk_index,
                    text=text,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=count_tokens(text),
                )
            )
    return chunks


def build_qrels(queries: list[QueryRecord], chunks: list[ChunkRecord]) -> list[QrelRecord]:
    chunks_by_context: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        chunks_by_context.setdefault(chunk.context_id, []).append(chunk)

    qrels: list[QrelRecord] = []
    seen: set[tuple[str, str]] = set()
    for query in queries:
        for chunk in chunks_by_context.get(query.context_id, []):
            match_type = ""
            if chunk.start_char >= 0 and overlaps(chunk.start_char, chunk.end_char, query.answer_start, query.answer_end):
                match_type = "span_overlap"
            elif query.answer_text.lower() in chunk.text.lower():
                match_type = "answer_text_contains"
            if not match_type:
                continue

            key = (query.query_id, chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            qrels.append(
                QrelRecord(
                    query_id=query.query_id,
                    chunk_id=chunk.chunk_id,
                    context_id=query.context_id,
                    relevance=1,
                    answer_text=query.answer_text,
                    answer_start=query.answer_start,
                    answer_end=query.answer_end,
                    match_type=match_type,
                )
            )
    return qrels


def group_qrels(qrels: list[QrelRecord]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for qrel in qrels:
        grouped.setdefault(qrel.query_id, set()).add(qrel.chunk_id)
    return grouped


# ---------------------------------------------------------------------------
# Retrieval, metrics, and IO
# ---------------------------------------------------------------------------


def retrieval_config(*, strategy: str, top_k: int, embedding_backend: str, embedding_model: str) -> dict[str, Any]:
    return {
        "retrieval": {"enabled": True, "strategy": strategy, "top_k": int(top_k)},
        "models": {
            "embedding_backend": embedding_backend,
            "embeddings": embedding_model,
            "sentence_transformer": embedding_model,
        },
    }


def chunks_to_documents(chunks: list[ChunkRecord]) -> list[Document]:
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


def normalize_retrieval_results(record: Any, chunk_lookup: dict[str, ChunkRecord], *, max_k: int) -> list[RetrievedItem]:
    items: list[RetrievedItem] = []
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
        if chunk is not None:
            text = text or chunk.text
            context_id = chunk.context_id
            token_count = chunk.token_count
        else:
            context_id = str(getattr(result, "document_id", ""))
            token_count = count_tokens(text)

        items.append(RetrievedItem(rank=len(items) + 1, chunk_id=chunk_id, context_id=context_id, text=text, token_count=token_count))
        if len(items) >= max_k:
            break
    return items


class RagasRunner:
    def __init__(self) -> None:
        self.precision_metric = NonLLMContextPrecisionWithReference()
        self.recall_metric = NonLLMContextRecall()

    def score(self, retrieved_contexts: list[str], reference_contexts: list[str]) -> tuple[float, float]:
        if not retrieved_contexts or not reference_contexts:
            return 0.0, 0.0
        sample = SingleTurnSample(retrieved_contexts=retrieved_contexts, reference_contexts=reference_contexts)
        return to_float(self.precision_metric.single_turn_score(sample)), to_float(self.recall_metric.single_turn_score(sample))


def ranking_metrics(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> dict[str, float]:
    top_ids = retrieved_ids[:k]
    return {
        "hit": 1.0 if any(chunk_id in relevant_ids for chunk_id in top_ids) else 0.0,
        "mrr": reciprocal_rank(top_ids, relevant_ids),
        "ndcg": ndcg(top_ids, relevant_ids, k),
    }


def aggregate_metrics(per_query: list[dict[str, Any]], *, retriever_build_ms: float, identifiers: dict[str, Any]) -> dict[str, Any]:
    if not per_query:
        row = {**identifiers, **{col: 0.0 for col in METRIC_COLUMNS}}
        row["retriever_build_ms"] = retriever_build_ms
        row["num_queries"] = 0
        return row
    df = pd.DataFrame(per_query)
    return {
        **identifiers,
        "ragas_precision": float(df["ragas_precision"].mean()),
        "ragas_recall": float(df["ragas_recall"].mean()),
        "ragas_f1": float(df["ragas_f1"].mean()),
        "ragas_f2": float(df["ragas_f2"].mean()),
        "hit": float(df["hit"].mean()),
        "mrr": float(df["mrr"].mean()),
        "ndcg": float(df["ndcg"].mean()),
        "context_tokens_mean": float(df["context_tokens"].mean()),
        "context_tokens_median": float(df["context_tokens"].median()),
        "context_tokens_p95": percentile(df["context_tokens"], 95),
        "latency_ms_mean": float(df["latency_ms"].mean()),
        "latency_ms_median": float(df["latency_ms"].median()),
        "latency_ms_p95": percentile(df["latency_ms"], 95),
        "retriever_build_ms": float(retriever_build_ms),
        "num_queries": int(len(df)),
    }


def write_inputs(out_dir: Path, contexts: list[ContextRecord], queries: list[QueryRecord], chunks: list[ChunkRecord], qrels: list[QrelRecord]) -> None:
    write_csv(out_dir / "contexts.csv", [asdict(x) for x in contexts])
    write_csv(out_dir / "queries.csv", [asdict(x) for x in queries])
    write_csv(out_dir / "qrels.csv", [asdict(x) for x in qrels])
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def make_context_id(*, title: str, text: str, index: int) -> str:
    return f"squad_{safe_id(title)}_{index:06d}_{stable_hash(text)[:10]}"


def safe_id(text: str) -> str:
    value = "".join(ch if ch.isalnum() else "_" for ch in str(text).lower())
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_")[:60] or "context"


def stable_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", errors="ignore")).hexdigest()


def locate_chunk(context: str, chunk: str, cursor: int) -> tuple[int, int]:
    idx = context.find(chunk, max(0, cursor - 50))
    if idx < 0:
        idx = context.find(chunk)
    if idx >= 0:
        return idx, idx + len(chunk)
    collapsed = " ".join(chunk.split())
    idx = context.find(collapsed, max(0, cursor - 50))
    if idx < 0:
        idx = context.find(collapsed)
    return (idx, idx + len(collapsed)) if idx >= 0 else (-1, -1)


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def count_tokens(text: str) -> int:
    return len(str(text).split())


def harmonic(precision: float, recall: float, *, beta: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    return 0.0 if denom <= 0.0 else (1 + beta2) * precision * recall / denom


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    gains = [1.0 if chunk_id in relevant_ids else 0.0 for chunk_id in retrieved_ids[:k]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_len = min(len(relevant_ids), k)
    if ideal_len == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_len))
    return dcg / idcg if idcg else 0.0


def percentile(values: Iterable[Any], p: float) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return 0.0 if arr.empty else float(np.percentile(arr, p))


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    return value


if __name__ == "__main__":
    main()

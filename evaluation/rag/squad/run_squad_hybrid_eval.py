#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
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


DEFAULT_BASE_FINAL_K_VALUES = [1, 3, 5, 10, 20, 50]
DEFAULT_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
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
class RetrievedItem:
    rank: int
    chunk_id: str
    context_id: str
    text: str
    token_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast production-hybrid SQuAD retrieval evaluation.")
    parser.add_argument("--input-dir", required=True, help="Directory containing chunks.jsonl, queries.csv, and qrels.csv")
    parser.add_argument("--out", default=None, help="Output directory. Default: <input-dir>/hybrid_fast")
    parser.add_argument("--bm25-k", type=int, required=True, help="BM25 candidate depth inside production hybrid retrieval")
    parser.add_argument("--vector-k", type=int, required=True, help="Vector candidate depth inside production hybrid retrieval")
    parser.add_argument(
        "--final-k-values",
        nargs="+",
        type=int,
        default=None,
        help="Final hybrid k values. Default: 1 3 5 10 20 50 N, where N=bm25_k+vector_k",
    )
    parser.add_argument("--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS, help="BM25 weights passed as hybrid_alpha")
    parser.add_argument("--embedding-backend", default="sentence_transformer")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--fail-on-empty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out) if args.out else input_dir / "hybrid_fast"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit").mkdir(parents=True, exist_ok=True)

    if args.bm25_k <= 0 or args.vector_k <= 0:
        raise ValueError("--bm25-k and --vector-k must be positive")

    final_k_values = resolve_final_k_values(args.final_k_values, bm25_k=args.bm25_k, vector_k=args.vector_k)
    max_final_k = max(final_k_values)
    weights = [float(w) for w in args.weights]

    print(f"Loading prepared SQuAD files from {input_dir}")
    chunks = read_chunks(input_dir / "chunks.jsonl")
    queries = read_queries(input_dir / "queries.csv")
    qrels_by_query = read_qrels(input_dir / "qrels.csv")
    queries = [query for query in queries if query.query_id in qrels_by_query]

    if args.fail_on_empty:
        if not chunks:
            raise ValueError("No chunks loaded")
        if not queries:
            raise ValueError("No usable queries loaded")
        if not qrels_by_query:
            raise ValueError("No qrels loaded")

    print(f"Chunks: {len(chunks)}")
    print(f"Queries: {len(queries)}")
    print(f"BM25 candidate depth: {args.bm25_k}")
    print(f"Vector candidate depth: {args.vector_k}")
    print(f"Fast hybrid mode: retrieve final top-{max_final_k} once per weight, evaluate prefixes {final_k_values}")

    documents = chunks_to_documents(chunks)
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    ragas_runner = RagasRunner()

    all_aggregate: list[dict[str, Any]] = []
    all_per_query: list[dict[str, Any]] = []

    for bm25_weight in weights:
        print(
            "Running production hybrid: "
            f"bm25_k={args.bm25_k}, vector_k={args.vector_k}, "
            f"top_k={max_final_k}, bm25_weight={bm25_weight}"
        )
        per_query, aggregate = evaluate_hybrid_prefixes(
            documents=documents,
            queries=queries,
            chunk_lookup=chunk_lookup,
            qrels_by_query=qrels_by_query,
            ragas_runner=ragas_runner,
            bm25_k=args.bm25_k,
            vector_k=args.vector_k,
            final_k_values=final_k_values,
            max_final_k=max_final_k,
            bm25_weight=bm25_weight,
            out_dir=out_dir,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
        )
        all_per_query.extend(per_query)
        all_aggregate.extend(aggregate)

    write_csv(out_dir / "hybrid_by_k.csv", all_aggregate)
    write_csv(out_dir / f"hybrid_bm25k{args.bm25_k}_vectork{args.vector_k}_by_k.csv", all_aggregate)
    write_csv(out_dir / "hybrid_per_query.csv", all_per_query)
    write_json(
        out_dir / "run_report.json",
        {
            "input_dir": str(input_dir),
            "bm25_k": args.bm25_k,
            "vector_k": args.vector_k,
            "final_k_values": final_k_values,
            "max_final_k_retrieved": max_final_k,
            "weights": weights,
            "num_chunks": len(chunks),
            "num_queries": len(queries),
            "metrics": METRIC_COLUMNS,
            "latency_policy": (
                "for each weight, production hybrid retrieval is executed once per query at max(final_k_values); "
                "latency_ms_* is top-max_final_k latency, while metrics are evaluated on top-k prefixes"
            ),
        },
    )
    print(f"Done. CSV files written to {out_dir}")


def evaluate_hybrid_prefixes(
    *,
    documents: list[Document],
    queries: list[QueryRecord],
    chunk_lookup: dict[str, ChunkRecord],
    qrels_by_query: dict[str, set[str]],
    ragas_runner: "RagasRunner",
    bm25_k: int,
    vector_k: int,
    final_k_values: list[int],
    max_final_k: int,
    bm25_weight: float,
    out_dir: Path,
    embedding_backend: str,
    embedding_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vector_weight = 1.0 - float(bm25_weight)
    run_id = f"hybrid_bm25k{bm25_k}_vectork{vector_k}_top{max_final_k}_bm25w{format_weight(bm25_weight)}_prefix"

    config = retrieval_config(
        bm25_k=bm25_k,
        vector_k=vector_k,
        final_k=max_final_k,
        bm25_weight=bm25_weight,
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

            retrieved = normalize_retrieval_results(record, chunk_lookup, max_k=max_final_k)
            audit_f.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "bm25_k": bm25_k,
                        "vector_k": vector_k,
                        "max_final_k_retrieved": max_final_k,
                        "bm25_weight": bm25_weight,
                        "vector_weight": vector_weight,
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

            for final_k in final_k_values:
                top = retrieved[:final_k]
                retrieved_ids = [item.chunk_id for item in top]
                retrieved_texts = [item.text for item in top]
                ragas_precision, ragas_recall = ragas_runner.score(retrieved_texts, reference_contexts)
                ranking = ranking_metrics(retrieved_ids, relevant_ids, final_k)
                per_query_rows.append(
                    {
                        "run_id": run_id,
                        "retriever": "hybrid",
                        "bm25_k": bm25_k,
                        "vector_k": vector_k,
                        "bm25_weight": bm25_weight,
                        "vector_weight": vector_weight,
                        "k": final_k,
                        "max_final_k_retrieved": max_final_k,
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

    aggregate_rows: list[dict[str, Any]] = []
    df = pd.DataFrame(per_query_rows)
    for final_k, group in df.groupby("k", dropna=False):
        aggregate_rows.append(
            aggregate_metrics(
                group.to_dict(orient="records"),
                retriever_build_ms=build_ms,
                identifiers={
                    "retriever": "hybrid",
                    "bm25_k": bm25_k,
                    "vector_k": vector_k,
                    "bm25_weight": bm25_weight,
                    "vector_weight": vector_weight,
                    "k": int(final_k),
                    "max_final_k_retrieved": max_final_k,
                },
            )
        )
    return per_query_rows, sorted(aggregate_rows, key=lambda row: row["k"])


def retrieval_config(
    *,
    bm25_k: int,
    vector_k: int,
    final_k: int,
    bm25_weight: float,
    embedding_backend: str,
    embedding_model: str,
) -> dict[str, Any]:
    return {
        "retrieval": {
            "enabled": True,
            "strategy": "hybrid",
            "top_k": int(final_k),
            "bm25_k": int(bm25_k),
            "vector_k": int(vector_k),
            "hybrid_alpha": float(bm25_weight),
        },
        "models": {
            "embedding_backend": embedding_backend,
            "embeddings": embedding_model,
            "sentence_transformer": embedding_model,
        },
    }


def resolve_final_k_values(explicit_values: list[int] | None, *, bm25_k: int, vector_k: int) -> list[int]:
    values = explicit_values if explicit_values else [*DEFAULT_BASE_FINAL_K_VALUES, bm25_k + vector_k]
    resolved = sorted({int(k) for k in values if 0 < int(k) <= bm25_k + vector_k})
    if not resolved:
        raise ValueError("At least one positive final k value is required")
    return resolved


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def read_chunks(path: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            chunks.append(
                ChunkRecord(
                    chunk_id=str(data["chunk_id"]),
                    context_id=str(data["context_id"]),
                    title=str(data.get("title", "")),
                    chunk_index=int(data.get("chunk_index", 0)),
                    text=str(data["text"]),
                    start_char=int(data.get("start_char", -1)),
                    end_char=int(data.get("end_char", -1)),
                    token_count=int(data.get("token_count", len(str(data["text"]).split()))),
                )
            )
    return chunks


def read_queries(path: Path) -> list[QueryRecord]:
    df = pd.read_csv(path)
    return [
        QueryRecord(
            query_id=str(row["query_id"]),
            context_id=str(row["context_id"]),
            title=str(row.get("title", "")),
            question=str(row["question"]),
            answer_text=str(row.get("answer_text", "")),
            answer_start=int(row.get("answer_start", -1)),
            answer_end=int(row.get("answer_end", -1)),
        )
        for row in df.to_dict(orient="records")
    ]


def read_qrels(path: Path) -> dict[str, set[str]]:
    df = pd.read_csv(path)
    grouped: dict[str, set[str]] = {}
    for row in df.to_dict(orient="records"):
        grouped.setdefault(str(row["query_id"]), set()).add(str(row["chunk_id"]))
    return grouped


# ---------------------------------------------------------------------------
# Retrieval, metrics, and IO
# ---------------------------------------------------------------------------


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


def format_weight(weight: float) -> str:
    return str(float(weight)).replace(".", "p")


if __name__ == "__main__":
    main()

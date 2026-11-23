#!/usr/bin/env python3
"""
SQuAD retrieval evaluation with Ragas non-LLM metrics and production RAG retrievers.

This is an updated version of the prototype experiment. Main fixes:

1. The reference contexts are the answer-bearing chunks only, not all chunks from
   the SQuAD paragraph.
2. Chunks keep stable metadata: chunk_id, context_id, offsets, token_count.
3. The script keeps many questions per context instead of one question per title.
4. BM25, vector, and hybrid retrieval are executed through the production
   retrieval interface. Hybrid experiments can use asymmetric BM25/vector
   candidate depths, e.g., BM25@100 and Vector@50, while still returning a
   controlled final top-k:

       from src.retrieval import build_retriever, retrieve_with_provenance

   The evaluator does not instantiate BM25Retriever, FAISS, or EnsembleRetriever
   directly.
5. Results are written as CSV and PDF figures.

Default outputs:
  <out>/contexts.csv
  <out>/chunks.jsonl
  <out>/queries.csv
  <out>/qrels.csv
  <out>/documents_preview.txt
  <out>/chunks_preview.txt
  <out>/bm25_by_k.csv
  <out>/vector_by_k.csv
  <out>/component_by_k.csv
  <out>/hybrid_weights_at_k<K>.csv
  <out>/per_query_metrics.csv
  <out>/best_config.json
  <out>/figures/bm25_vector_by_k.pdf
  <out>/figures/hybrid_weights_at_k<K>.pdf
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
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
except Exception:  # pragma: no cover - depends on ragas version
    from ragas.dataset_schema import SingleTurnSample

from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall


DEFAULT_K_VALUES = [1, 5, 10, 20, 30, 40, 50]
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate production BM25/vector/hybrid RAG retrievers on SQuAD answer-bearing chunks."
    )
    parser.add_argument("--dataset", default="rajpurkar/squad", help="Hugging Face dataset name. Default: rajpurkar/squad")
    parser.add_argument("--split", default="train", help="Dataset split. Default: train")
    parser.add_argument("--out", default="evaluation/rag/results/squad_ragas", help="Output directory")
    parser.add_argument("--limit-contexts", type=int, default=None, help="Limit number of unique contexts for smoke tests")
    parser.add_argument("--limit-questions", type=int, default=None, help="Limit number of questions after context filtering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for deterministic shuffling")
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--max-k", type=int, default=None, help="Retrieval depth used internally. Defaults to max(k-values)")
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "vector"], choices=["bm25", "vector"])
    parser.add_argument("--hybrid-weights", nargs="+", type=float, default=DEFAULT_HYBRID_WEIGHTS)
    parser.add_argument("--skip-hybrid", action="store_true")
    parser.add_argument(
        "--fixed-k",
        type=int,
        default=None,
        help=(
            "Backward-compatible alias for --hybrid-final-k. "
            "It controls the final number of chunks returned by the production hybrid retriever."
        ),
    )
    parser.add_argument(
        "--hybrid-bm25-k",
        type=int,
        default=None,
        help=(
            "Candidate depth for the BM25 branch inside the production hybrid retriever. "
            "Example: --hybrid-bm25-k 100. Defaults to the final hybrid k if omitted."
        ),
    )
    parser.add_argument(
        "--hybrid-vector-k",
        type=int,
        default=None,
        help=(
            "Candidate depth for the vector branch inside the production hybrid retriever. "
            "Example: --hybrid-vector-k 50. Defaults to the final hybrid k if omitted."
        ),
    )
    parser.add_argument(
        "--hybrid-final-k",
        type=int,
        default=None,
        help=(
            "Final top-k returned after the production hybrid retriever merges BM25 and vector results. "
            "If omitted, the script uses --fixed-k, or the maximum selected component k."
        ),
    )
    parser.add_argument(
        "--hybrid-final-k-values",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Evaluate several final top-k values for the production hybrid retriever. "
            "This overrides --hybrid-final-k/--fixed-k. Example: --hybrid-final-k-values 5 10 20 30 50."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        default="ragas_f1",
        choices=["ragas_f1", "ragas_f2", "exact_f1", "exact_f2", "ragas_recall", "exact_recall", "mrr", "ndcg"],
        help="Metric used to select best k for BM25/vector. Default: ragas_f1",
    )
    parser.add_argument("--chunk-size", type=int, default=200, help="ClusterSemanticChunker max_chunk_size. Default: 200")
    parser.add_argument("--embedding-backend", default="sentence_transformer")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--no-ragas", action="store_true", help="Skip Ragas metrics and use exact chunk-id metrics only")
    parser.add_argument("--fail-on-empty", action="store_true")
    parser.add_argument(
        "--preview-count",
        type=int,
        default=5,
        help="Number of source documents and chunks to print as a console/file preview. Default: 5",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable console/file previews of loaded documents and generated chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    k_values = sorted({int(k) for k in args.k_values})
    max_k = int(args.max_k or max(k_values))
    k_values = [k for k in k_values if k <= max_k]

    print(f"Loading dataset {args.dataset!r}, split={args.split!r}")
    ds = load_dataset(args.dataset, split=args.split)
    contexts, queries = build_contexts_and_queries(ds, limit_contexts=args.limit_contexts, limit_questions=args.limit_questions)

    print(f"Unique contexts: {len(contexts)}")
    print(f"Questions before qrel filtering: {len(queries)}")

    # Show a small, deterministic preview of the source documents that will be
    # converted into retriever documents. This is useful when validating smoke
    # runs, because SQuAD contains repeated titles with many distinct paragraphs.
    if not args.no_preview:
        print_context_preview(contexts, preview_count=args.preview_count, out_dir=out_dir)

    print("Chunking contexts")
    chunks = chunk_contexts(contexts, max_chunk_size=args.chunk_size)
    qrels = build_qrels(queries, chunks)
    qrels_by_query = group_qrels(qrels)
    queries = [q for q in queries if q.query_id in qrels_by_query]

    if args.fail_on_empty:
        if not contexts:
            raise ValueError("No contexts loaded")
        if not chunks:
            raise ValueError("No chunks produced")
        if not queries:
            raise ValueError("No questions with usable qrels")
        if not qrels:
            raise ValueError("No qrels produced")

    print(f"Chunks: {len(chunks)}")
    print(f"Usable questions with answer-bearing chunk qrels: {len(queries)}")
    print(f"Qrels: {len(qrels)}")

    # Print the actual generated chunks after qrels are built. The same chunk
    # objects are later converted to LangChain Documents and passed to the
    # production retriever, so this preview is the easiest place to inspect the
    # evaluation corpus before retrieval begins.
    if not args.no_preview:
        print_chunk_preview(chunks, preview_count=args.preview_count, out_dir=out_dir)

    write_inputs(out_dir, contexts, queries, chunks, qrels)
    write_json(
        out_dir / "load_report.json",
        {
            "dataset": args.dataset,
            "split": args.split,
            "num_contexts": len(contexts),
            "num_queries": len(queries),
            "num_chunks": len(chunks),
            "num_qrels": len(qrels),
            "chunker": "ClusterSemanticChunker",
            "chunk_size": args.chunk_size,
            "k_values": k_values,
            "max_k": max_k,
            "retriever_component_source": "src.retrieval.build_retriever and src.retrieval.retrieve_with_provenance",
            "reference_context_policy": "answer-bearing chunks only",
            "ragas_enabled": not args.no_ragas,
        },
    )

    documents = chunks_to_documents(chunks)
    if not args.no_preview:
        print_retriever_document_preview(documents, preview_count=args.preview_count, out_dir=out_dir)
    chunk_lookup = {c.chunk_id: c for c in chunks}
    query_lookup = {q.query_id: q for q in queries}
    ragas_metrics = RagasMetrics(enabled=not args.no_ragas)

    all_per_query: list[dict[str, Any]] = []
    all_aggregate: list[dict[str, Any]] = []
    component_aggregate: list[dict[str, Any]] = []

    for retriever in args.retrievers:
        run_id = f"{retriever}_top{max_k}"
        print(f"Running {retriever} retrieval at max_k={max_k}")
        per_query, aggregate = run_retriever(
            retriever=retriever,
            run_id=run_id,
            documents=documents,
            chunk_lookup=chunk_lookup,
            query_lookup=query_lookup,
            queries=queries,
            qrels_by_query=qrels_by_query,
            ragas_metrics=ragas_metrics,
            k_values=k_values,
            max_k=max_k,
            out_dir=out_dir,
            args=args,
        )
        all_per_query.extend(per_query)
        all_aggregate.extend(aggregate)
        component_aggregate.extend(aggregate)
        pd.DataFrame(aggregate).to_csv(out_dir / f"{retriever}_by_k.csv", index=False)

    component_df = pd.DataFrame(component_aggregate)
    component_df.to_csv(out_dir / "component_by_k.csv", index=False)
    pd.DataFrame(all_per_query).to_csv(out_dir / "per_query_metrics.csv", index=False)
    pd.DataFrame(all_aggregate).to_csv(out_dir / "aggregate_metrics.csv", index=False)

    selected = select_best_k(component_df, args.selection_metric)
    write_json(out_dir / "selected_k.json", selected)

    selected_ks = [int(v["selected_k"]) for v in selected.values() if "selected_k" in v]

    # Production hybrid execution supports asymmetric candidate depths and, now,
    # a sweep over several final k values.  Each tested configuration is executed
    # through src.retrieval.build_retriever(strategy="hybrid"), so this is the
    # same path used by the pipeline rather than cached rank fusion.
    hybrid_final_ks = resolve_hybrid_final_ks(args, selected_ks=selected_ks, fallback_k=max_k)

    hybrid_rows: list[dict[str, Any]] = []
    if not args.skip_hybrid:
        print(
            "Running production hybrid retrieval with "
            f"final_k_values={hybrid_final_ks}, "
            f"bm25_k={args.hybrid_bm25_k or 'same as final_k'}, "
            f"vector_k={args.hybrid_vector_k or 'same as final_k'}"
        )
        for hybrid_k in hybrid_final_ks:
            # If branch-specific candidate depths are not provided, each branch
            # defaults to the tested final k.  If provided, the same asymmetric
            # candidate depths are held fixed while final k varies.
            hybrid_bm25_k = int(args.hybrid_bm25_k or hybrid_k)
            hybrid_vector_k = int(args.hybrid_vector_k or hybrid_k)

            for weight in args.hybrid_weights:
                run_id = (
                    f"hybrid_bm25k{hybrid_bm25_k}_vectork{hybrid_vector_k}"
                    f"_finalk{hybrid_k}_bm25w_{format_weight(weight)}"
                )
                print(
                    "Running production hybrid retrieval: "
                    f"bm25_weight={weight}, vector_weight={1.0 - float(weight):.3f}, "
                    f"bm25_k={hybrid_bm25_k}, vector_k={hybrid_vector_k}, final_k={hybrid_k}"
                )
                per_query, aggregate = run_retriever(
                    retriever="hybrid",
                    run_id=run_id,
                    documents=documents,
                    chunk_lookup=chunk_lookup,
                    query_lookup=query_lookup,
                    queries=queries,
                    qrels_by_query=qrels_by_query,
                    ragas_metrics=ragas_metrics,
                    k_values=[hybrid_k],
                    max_k=hybrid_k,
                    out_dir=out_dir,
                    args=args,
                    bm25_weight=float(weight),
                    hybrid_bm25_k=hybrid_bm25_k,
                    hybrid_vector_k=hybrid_vector_k,
                )
                all_per_query.extend(per_query)
                all_aggregate.extend(aggregate)
                hybrid_rows.extend(aggregate)

        hybrid_df = pd.DataFrame(hybrid_rows)
        hybrid_df.to_csv(out_dir / "hybrid_by_final_k.csv", index=False)

        # Stable, descriptive filename for the common asymmetric setup where
        # branch depths are fixed and only final k/weight vary.
        if args.hybrid_bm25_k is not None and args.hybrid_vector_k is not None:
            hybrid_df.to_csv(
                out_dir / f"hybrid_bm25k{int(args.hybrid_bm25_k)}_vectork{int(args.hybrid_vector_k)}_by_final_k.csv",
                index=False,
            )

        # Backward-compatible per-final-k files expected by older inspection
        # commands, e.g., hybrid_weights_at_k30.csv.
        if not hybrid_df.empty and "final_k" in hybrid_df.columns:
            for final_k, group in hybrid_df.groupby("final_k", dropna=False):
                group.to_csv(out_dir / f"hybrid_weights_at_k{int(final_k)}.csv", index=False)

        pd.DataFrame(all_per_query).to_csv(out_dir / "per_query_metrics.csv", index=False)
        pd.DataFrame(all_aggregate).to_csv(out_dir / "aggregate_metrics.csv", index=False)

    plot_component_curves(component_df, out_dir / "figures" / "bm25_vector_by_k.pdf")
    plot_individual_retriever_curves(component_df, out_dir / "figures")
    hybrid_df = pd.DataFrame(hybrid_rows)
    if hybrid_rows:
        for final_k, group in hybrid_df.groupby("final_k", dropna=False):
            plot_hybrid_weights(group, int(final_k), out_dir / "figures" / f"hybrid_weights_at_k{int(final_k)}.pdf")
        plot_hybrid_by_final_k(
            hybrid_df,
            selection_metric=args.selection_metric,
            path=out_dir / "figures" / "hybrid_by_final_k.pdf",
        )

    best_config = build_best_config(
        selected,
        hybrid_df,
        hybrid_final_ks,
        args.selection_metric,
        requested_hybrid_bm25_k=args.hybrid_bm25_k,
        requested_hybrid_vector_k=args.hybrid_vector_k,
    )
    write_json(out_dir / "best_config.json", best_config)
    write_json(
        out_dir / "final_report.json",
        {
            "selected_k_by_component": selected,
            "hybrid_final_k_values": hybrid_final_ks,
            "requested_hybrid_bm25_k": args.hybrid_bm25_k,
            "requested_hybrid_vector_k": args.hybrid_vector_k,
            "hybrid_execution": "production_pipeline_hybrid",
            "best_config": best_config,
            "outputs": {
                "component_by_k_csv": str(out_dir / "component_by_k.csv"),
                "hybrid_by_final_k_csv": str(out_dir / "hybrid_by_final_k.csv"),
                "per_query_metrics_csv": str(out_dir / "per_query_metrics.csv"),
                "aggregate_metrics_csv": str(out_dir / "aggregate_metrics.csv"),
                "best_config_json": str(out_dir / "best_config.json"),
                "combined_component_figure": str(out_dir / "figures" / "bm25_vector_by_k.pdf"),
                "bm25_figure": str(out_dir / "figures" / "bm25_by_k.pdf"),
                "vector_figure": str(out_dir / "figures" / "vector_by_k.pdf"),
                "hybrid_by_final_k_figure": str(out_dir / "figures" / "hybrid_by_final_k.pdf"),
            },
        },
    )
    print(f"Done. Outputs written to {out_dir}")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def build_contexts_and_queries(ds: Dataset, *, limit_contexts: int | None, limit_questions: int | None) -> tuple[list[ContextRecord], list[QueryRecord]]:
    """Collect unique SQuAD paragraphs and all associated QA pairs.

    SQuAD has many questions for the same paragraph. We keep a single
    ContextRecord per unique paragraph and one QueryRecord per QA item. This
    avoids the earlier one-question-per-title problem while preserving the
    many-questions-per-context structure needed for retrieval evaluation.
    """
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
    """Chunk every source paragraph and preserve character offsets.

    Ragas can compare retrieved/reference texts, but the exact evaluation also
    needs stable chunk IDs and answer-span overlap. Therefore, every generated
    chunk stores the parent context ID and best-effort character offsets in the
    original SQuAD paragraph.
    """
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    chunker = ClusterSemanticChunker(default_ef, max_chunk_size=max_chunk_size)
    chunks: list[ChunkRecord] = []

    for ctx in tqdm(contexts, desc="Chunking contexts"):
        chunk_texts = chunker.split_text(ctx.text)
        cursor = 0
        for idx, raw_chunk in enumerate(chunk_texts, start=1):
            text = str(raw_chunk).strip()
            if not text:
                continue
            start, end = locate_chunk(ctx.text, text, cursor)
            if start >= 0:
                cursor = max(cursor, end)
            chunk_id = f"{ctx.context_id}_c{idx:05d}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    context_id=ctx.context_id,
                    title=ctx.title,
                    chunk_index=idx,
                    text=text,
                    start_char=start,
                    end_char=end,
                    token_count=count_tokens(text),
                )
            )
    return chunks


def build_qrels(queries: list[QueryRecord], chunks: list[ChunkRecord]) -> list[QrelRecord]:
    """Create answer-span-grounded qrels.

    A chunk is relevant only when it overlaps the gold SQuAD answer span. If
    offset matching fails because the semantic chunker normalized whitespace, we
    fall back to a case-insensitive answer-text containment check.
    """
    chunks_by_context: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        chunks_by_context.setdefault(chunk.context_id, []).append(chunk)

    qrels: list[QrelRecord] = []
    seen: set[tuple[str, str]] = set()
    for query in queries:
        for chunk in chunks_by_context.get(query.context_id, []):
            match_type = ""
            if chunk.start_char >= 0 and chunk.end_char >= 0 and overlaps(chunk.start_char, chunk.end_char, query.answer_start, query.answer_end):
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


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def chunks_to_documents(chunks: list[ChunkRecord]) -> list[Document]:
    """Convert internal chunk records to LangChain Documents for the pipeline.

    The metadata keys intentionally include both `chunk_id` and `document_id`
    because the production retriever/provenance layer uses them to reconstruct
    which retrieved chunk came from which source context.
    """
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


def group_qrels(qrels: list[QrelRecord]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for qrel in qrels:
        out.setdefault(qrel.query_id, set()).add(qrel.chunk_id)
    return out


def run_retriever(
    *,
    retriever: str,
    run_id: str,
    documents: list[Document],
    chunk_lookup: dict[str, ChunkRecord],
    query_lookup: dict[str, QueryRecord],
    queries: list[QueryRecord],
    qrels_by_query: dict[str, set[str]],
    ragas_metrics: "RagasMetrics",
    k_values: list[int],
    max_k: int,
    out_dir: Path,
    args: argparse.Namespace,
    bm25_weight: float | None = None,
    hybrid_bm25_k: int | None = None,
    hybrid_vector_k: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one retriever configuration once at max_k, then rescore cutoffs.

    The retriever is built with top_k=max_k so each query is executed only one
    time per configuration. We then evaluate prefixes at k=1,3,5,... from the
    same ranked list. This keeps BM25/vector/hybrid comparisons consistent and
    avoids repeatedly rebuilding indices.
    """
    config = retrieval_config(
        strategy=retriever,
        top_k=max_k,
        bm25_weight=bm25_weight,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        hybrid_bm25_k=hybrid_bm25_k,
        hybrid_vector_k=hybrid_vector_k,
    )

    build_start = time.perf_counter()
    bundle = build_retriever(documents, config)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    audit_path = out_dir / "audit" / f"{run_id}_retrieved_chunks.jsonl"
    per_query: list[dict[str, Any]] = []

    with audit_path.open("w", encoding="utf-8") as audit_f:
        for query in tqdm(queries, desc=f"Retrieving {run_id}"):
            relevant_ids = qrels_by_query.get(query.query_id, set())
            if not relevant_ids:
                continue

            start = time.perf_counter()
            record = retrieve_with_provenance(
                bundle,
                query=query.question,
                run_id=run_id,
                case_id="squad",
                attack_step_id=query.query_id,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            retrieved = normalize_retrieval_results(record, chunk_lookup, max_k=max_k)
            reference_contexts = [chunk_lookup[cid].text for cid in sorted(relevant_ids) if cid in chunk_lookup]

            audit_f.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "retriever": retriever,
                        "query_id": query.query_id,
                        "question": query.question,
                        "answer_text": query.answer_text,
                        "gold_chunk_ids": sorted(relevant_ids),
                        "bm25_weight": bm25_weight,
                        "vector_weight": None if bm25_weight is None else 1.0 - float(bm25_weight),
                        "bm25_k": hybrid_bm25_k,
                        "vector_k": hybrid_vector_k,
                        "final_k": max_k,
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
                exact = exact_metrics(retrieved_ids, relevant_ids, k)
                ragas_p, ragas_r = ragas_metrics.score(retrieved_texts, reference_contexts)
                ragas_f1 = harmonic(ragas_p, ragas_r, beta=1.0)
                ragas_f2 = harmonic(ragas_p, ragas_r, beta=2.0)

                per_query.append(
                    {
                        "run_id": run_id,
                        "retriever": retriever,
                        "bm25_weight": bm25_weight,
                        "vector_weight": None if bm25_weight is None else 1.0 - float(bm25_weight),
                        "bm25_k": hybrid_bm25_k,
                        "vector_k": hybrid_vector_k,
                        "final_k": max_k if retriever == "hybrid" else None,
                        "k": k,
                        "query_id": query.query_id,
                        "context_id": query.context_id,
                        "exact_precision": exact["precision"],
                        "exact_recall": exact["recall"],
                        "exact_f1": exact["f1"],
                        "exact_f2": exact["f2"],
                        "ragas_precision": ragas_p,
                        "ragas_recall": ragas_r,
                        "ragas_f1": ragas_f1,
                        "ragas_f2": ragas_f2,
                        "hit": exact["hit"],
                        "mrr": exact["mrr"],
                        "ndcg": exact["ndcg"],
                        "context_tokens": sum(item.token_count for item in top),
                        "latency_ms": latency_ms,
                        "num_relevant_chunks": len(relevant_ids),
                        "num_retrieved_chunks": len(top),
                    }
                )

    aggregate = aggregate_metrics(per_query, build_ms=build_ms)
    return per_query, aggregate


def retrieval_config(
    *,
    strategy: str,
    top_k: int,
    bm25_weight: float | None,
    embedding_backend: str,
    embedding_model: str,
    hybrid_bm25_k: int | None = None,
    hybrid_vector_k: int | None = None,
) -> dict[str, Any]:
    # top_k is the final number of chunks returned by retrieve_with_provenance.
    # For hybrid retrieval, bm25_k/vector_k are separate candidate depths for
    # the production BM25 and vector branches.  This lets us evaluate, for
    # example, BM25@100 + Vector@50 while still returning a smaller final top-k.
    retrieval: dict[str, Any] = {"enabled": True, "strategy": strategy, "top_k": top_k}
    if strategy == "hybrid":
        weight = 0.5 if bm25_weight is None else float(bm25_weight)
        retrieval.update(
            {
                "bm25_k": int(hybrid_bm25_k or top_k),
                "vector_k": int(hybrid_vector_k or top_k),
                "hybrid_alpha": weight,
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
    items: list[RetrievedItem] = []
    seen: set[str] = set()
    raw_results = getattr(record, "results", [])

    for idx, result in enumerate(raw_results, start=1):
        chunk_id = str(getattr(result, "chunk_id", "") or "")
        text = str(getattr(result, "text", "") or "")
        if not chunk_id:
            metadata = getattr(result, "metadata", {}) or {}
            chunk_id = str(metadata.get("chunk_id", ""))
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunk = chunk_lookup.get(chunk_id)
        context_id = chunk.context_id if chunk else str(getattr(result, "document_id", ""))
        token_count = chunk.token_count if chunk else count_tokens(text)
        if chunk and not text:
            text = chunk.text
        items.append(RetrievedItem(rank=len(items) + 1, chunk_id=chunk_id, context_id=context_id, text=text, token_count=token_count))
        if len(items) >= max_k:
            break
    return items


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class RagasMetrics:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        if enabled:
            self.precision_metric = NonLLMContextPrecisionWithReference()
            self.recall_metric = NonLLMContextRecall()
        else:
            self.precision_metric = None
            self.recall_metric = None

    def score(self, retrieved_contexts: list[str], reference_contexts: list[str]) -> tuple[float, float]:
        if not self.enabled:
            return 0.0, 0.0
        if not reference_contexts:
            return 0.0, 0.0
        sample = SingleTurnSample(retrieved_contexts=retrieved_contexts, reference_contexts=reference_contexts)
        precision = to_float(self.precision_metric.single_turn_score(sample))
        recall = to_float(self.recall_metric.single_turn_score(sample))
        return precision, recall


def exact_metrics(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> dict[str, float]:
    top_ids = retrieved_ids[:k]
    rel_count = len(set(top_ids) & relevant_ids)
    precision = rel_count / k if k else 0.0
    recall = rel_count / len(relevant_ids) if relevant_ids else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": harmonic(precision, recall, beta=1.0),
        "f2": harmonic(precision, recall, beta=2.0),
        "hit": 1.0 if rel_count > 0 else 0.0,
        "mrr": reciprocal_rank(top_ids, relevant_ids),
        "ndcg": ndcg(top_ids, relevant_ids, k),
    }


def aggregate_metrics(per_query: list[dict[str, Any]], *, build_ms: float) -> list[dict[str, Any]]:
    if not per_query:
        return []
    df = pd.DataFrame(per_query)
    rows: list[dict[str, Any]] = []
    group_cols = ["run_id", "retriever", "bm25_weight", "vector_weight", "bm25_k", "vector_k", "final_k", "k"]
    for keys, group in df.groupby(group_cols, dropna=False):
        run_id, retriever, bm25_weight, vector_weight, bm25_k, vector_k, final_k, k = keys
        row = {
            "run_id": run_id,
            "retriever": retriever,
            "bm25_weight": none_if_nan(bm25_weight),
            "vector_weight": none_if_nan(vector_weight),
            "bm25_k": none_if_nan(bm25_k),
            "vector_k": none_if_nan(vector_k),
            "final_k": none_if_nan(final_k),
            "k": int(k),
            "exact_precision": float(group["exact_precision"].mean()),
            "exact_recall": float(group["exact_recall"].mean()),
            "exact_f1": float(group["exact_f1"].mean()),
            "exact_f2": float(group["exact_f2"].mean()),
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
            "retriever_build_ms": build_ms,
            "num_queries": int(len(group)),
        }
        rows.append(row)
    return rows


def select_best_k(df: pd.DataFrame, selection_metric: str) -> dict[str, Any]:
    if df.empty:
        return {}
    output: dict[str, Any] = {}
    for retriever, group in df.groupby("retriever", dropna=False):
        sort_cols = [selection_metric, "mrr", "ndcg", "context_tokens_mean", "latency_ms_mean", "k"]
        ascending = [False, False, False, True, True, True]
        ranked = group.sort_values(sort_cols, ascending=ascending, na_position="last")
        row = ranked.iloc[0].to_dict()
        output[str(retriever)] = {
            "selection_metric": selection_metric,
            "selected_k": int(row["k"]),
            "metrics": jsonable(row),
        }
    return output


def resolve_hybrid_final_ks(args: argparse.Namespace, *, selected_ks: list[int], fallback_k: int) -> list[int]:
    """Resolve the list of final k values to test for production hybrid retrieval.

    The component BM25/vector curves are prefix evaluations over max_k.  Hybrid
    evaluation is different: each final k is executed as an actual pipeline
    configuration with retrieval.strategy="hybrid" and retrieval.top_k=final_k.
    """
    if args.hybrid_final_k_values:
        values = args.hybrid_final_k_values
    elif args.hybrid_final_k is not None:
        values = [args.hybrid_final_k]
    elif args.fixed_k is not None:
        values = [args.fixed_k]
    elif selected_ks:
        values = [max(selected_ks)]
    else:
        values = [fallback_k]

    resolved = sorted({int(k) for k in values if int(k) > 0})
    if not resolved:
        raise ValueError("At least one positive hybrid final k must be provided")
    return resolved


def build_best_config(
    selected: dict[str, Any],
    hybrid_df: pd.DataFrame,
    hybrid_final_ks: list[int],
    selection_metric: str,
    *,
    requested_hybrid_bm25_k: int | None = None,
    requested_hybrid_vector_k: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "selection_metric": selection_metric,
        "selected_k_by_component": selected,
        "hybrid_execution": "production_pipeline_hybrid",
        "hybrid_final_k_values": hybrid_final_ks,
        "requested_hybrid_bm25_k": requested_hybrid_bm25_k,
        "requested_hybrid_vector_k": requested_hybrid_vector_k,
    }
    if not hybrid_df.empty:
        ranked = hybrid_df.sort_values(
            [selection_metric, "mrr", "ndcg", "context_tokens_mean", "latency_ms_mean", "final_k"],
            ascending=[False, False, False, True, True, True],
            na_position="last",
        )
        payload["best_hybrid"] = jsonable(ranked.iloc[0].to_dict())

        best_by_final_k: dict[str, Any] = {}
        for final_k, group in hybrid_df.groupby("final_k", dropna=False):
            ranked_group = group.sort_values(
                [selection_metric, "mrr", "ndcg", "context_tokens_mean", "latency_ms_mean"],
                ascending=[False, False, False, True, True],
                na_position="last",
            )
            best_by_final_k[str(int(final_k))] = jsonable(ranked_group.iloc[0].to_dict())
        payload["best_hybrid_by_final_k"] = best_by_final_k
    return payload


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_component_curves(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for retriever, group in df.groupby("retriever"):
        group = group.sort_values("k")
        label = str(retriever).upper()
        ax.plot(group["k"], group["ragas_precision"], marker="o", linestyle=":", label=f"{label} Ragas precision")
        ax.plot(group["k"], group["ragas_recall"], marker="^", linestyle="--", label=f"{label} Ragas recall")
        ax.plot(group["k"], group["ragas_f1"], marker="s", linestyle="-", label=f"{label} Ragas F1")
    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    ax.set_title("SQuAD answer-chunk retrieval using Ragas non-LLM metrics")
    ax.set_xticks(sorted(df["k"].unique()))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_individual_retriever_curves(df: pd.DataFrame, figures_dir: Path) -> None:
    """Write one standalone retrieval curve per component.

    The combined BM25/vector figure is useful for quick comparison, but the
    separate figures are easier to use in the paper because each plot has only
    one retriever and a compact legend.
    """
    if df.empty:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    for retriever, group in df.groupby("retriever"):
        retriever_name = str(retriever).lower()
        if retriever_name not in {"bm25", "vector"}:
            continue
        plot_single_retriever_curve(
            group.sort_values("k"),
            title=f"SQuAD answer-chunk retrieval: {retriever_name.upper()}",
            path=figures_dir / f"{retriever_name}_by_k.pdf",
        )


def plot_single_retriever_curve(df: pd.DataFrame, *, title: str, path: Path) -> None:
    if df.empty:
        return
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(df["k"], df["ragas_precision"], marker="o", linestyle=":", label="Ragas precision")
    ax.plot(df["k"], df["ragas_recall"], marker="^", linestyle="--", label="Ragas recall")
    ax.plot(df["k"], df["ragas_f1"], marker="s", linestyle="-", label="Ragas F1")
    ax.plot(df["k"], df["ragas_f2"], marker="D", linestyle="-.", label="Ragas F2")
    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(sorted(df["k"].unique()))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_hybrid_weights(df: pd.DataFrame, hybrid_k: int, path: Path) -> None:
    if df.empty:
        return
    df = df.sort_values("bm25_weight")
    bm25_k = first_non_null(df.get("bm25_k"))
    vector_k = first_non_null(df.get("vector_k"))
    title_suffix = f"BM25@{int(bm25_k)} + Vector@{int(vector_k)}, final k={hybrid_k}" if bm25_k is not None and vector_k is not None else f"final k={hybrid_k}"
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(df["bm25_weight"], df["ragas_precision"], marker="o", linestyle=":", label="Ragas precision")
    ax.plot(df["bm25_weight"], df["ragas_recall"], marker="^", linestyle="--", label="Ragas recall")
    ax.plot(df["bm25_weight"], df["ragas_f1"], marker="s", linestyle="-", label="Ragas F1")
    ax.plot(df["bm25_weight"], df["ragas_f2"], marker="D", linestyle="-.", label="Ragas F2")
    ax.set_xlabel("BM25 weight in hybrid retriever")
    ax.set_ylabel("Score")
    ax.set_title(f"SQuAD production hybrid retrieval ({title_suffix})")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_hybrid_by_final_k(df: pd.DataFrame, *, selection_metric: str, path: Path) -> None:
    """Plot the selected hybrid metric as final_k varies.

    Each line corresponds to one BM25/vector weight.  This figure is useful for
    choosing the final context depth after fixing the branch candidate pools
    such as BM25@100 and Vector@50.
    """
    if df.empty or "final_k" not in df.columns or selection_metric not in df.columns:
        return
    df = df.sort_values(["bm25_weight", "final_k"])
    bm25_k = first_non_null(df.get("bm25_k"))
    vector_k = first_non_null(df.get("vector_k"))
    title_suffix = f"BM25@{int(bm25_k)} + Vector@{int(vector_k)}" if bm25_k is not None and vector_k is not None else "production hybrid"

    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    for weight, group in df.groupby("bm25_weight", dropna=False):
        group = group.sort_values("final_k")
        ax.plot(
            group["final_k"],
            group[selection_metric],
            marker="o",
            linestyle="-",
            label=f"BM25 weight={float(weight):.2f}",
        )
    ax.set_xlabel("Final hybrid k")
    ax.set_ylabel(selection_metric)
    ax.set_title(f"SQuAD production hybrid retrieval by final k ({title_suffix})")
    ax.set_xticks(sorted(int(k) for k in df["final_k"].dropna().unique()))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console/file previews
# ---------------------------------------------------------------------------


def print_context_preview(contexts: list[ContextRecord], *, preview_count: int, out_dir: Path) -> None:
    """Print and persist a compact preview of loaded source documents."""
    rows = []
    header = f"Available source documents/contexts: {len(contexts)}"
    rows.append(header)
    rows.append("-" * len(header))
    for ctx in contexts[: max(0, preview_count)]:
        rows.append(f"context_id={ctx.context_id}")
        rows.append(f"title={ctx.title}")
        rows.append(f"source_index={ctx.source_index}, chars={len(ctx.text)}, tokens={count_tokens(ctx.text)}")
        rows.append(f"text_preview={truncate_text(ctx.text, 280)}")
        rows.append("")
    message = "\n".join(rows).rstrip()
    print("\n" + message + "\n")
    (out_dir / "documents_preview.txt").write_text(message + "\n", encoding="utf-8")


def print_chunk_preview(chunks: list[ChunkRecord], *, preview_count: int, out_dir: Path) -> None:
    """Print and persist a compact preview of generated chunks."""
    rows = []
    header = f"Generated chunks: {len(chunks)}"
    rows.append(header)
    rows.append("-" * len(header))
    for chunk in chunks[: max(0, preview_count)]:
        rows.append(f"chunk_id={chunk.chunk_id}")
        rows.append(f"context_id={chunk.context_id}")
        rows.append(
            f"chunk_index={chunk.chunk_index}, chars=[{chunk.start_char},{chunk.end_char}), "
            f"tokens={chunk.token_count}"
        )
        rows.append(f"text_preview={truncate_text(chunk.text, 280)}")
        rows.append("")
    message = "\n".join(rows).rstrip()
    print("\n" + message + "\n")
    (out_dir / "chunks_preview.txt").write_text(message + "\n", encoding="utf-8")


def print_retriever_document_preview(documents: list[Document], *, preview_count: int, out_dir: Path) -> None:
    """Print a preview of the LangChain Documents passed to build_retriever."""
    rows = []
    header = f"Retriever documents available to the pipeline: {len(documents)}"
    rows.append(header)
    rows.append("-" * len(header))
    for doc in documents[: max(0, preview_count)]:
        meta = doc.metadata or {}
        rows.append(f"chunk_id={meta.get('chunk_id')}")
        rows.append(f"document_id={meta.get('document_id')} title={meta.get('title')}")
        rows.append(
            f"chunk_index={meta.get('chunk_index')}, "
            f"chars=[{meta.get('start_char')},{meta.get('end_char')}), "
            f"tokens={meta.get('token_count')}"
        )
        rows.append(f"text_preview={truncate_text(doc.page_content, 280)}")
        rows.append("")
    message = "\n".join(rows).rstrip()
    print("\n" + message + "\n")
    (out_dir / "retriever_documents_preview.txt").write_text(message + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# IO and utilities
# ---------------------------------------------------------------------------


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def make_context_id(*, title: str, text: str, index: int) -> str:
    return f"squad_{safe_id(title)}_{index:06d}_{stable_hash(text)[:10]}"


def safe_id(text: str) -> str:
    chars = [ch if ch.isalnum() else "_" for ch in str(text).lower()]
    value = "".join(chars)
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_")[:60] or "context"


def stable_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", errors="ignore")).hexdigest()


def locate_chunk(context: str, chunk: str, cursor: int) -> tuple[int, int]:
    if not chunk:
        return -1, -1
    idx = context.find(chunk, max(0, cursor - 50))
    if idx < 0:
        idx = context.find(chunk)
    if idx >= 0:
        return idx, idx + len(chunk)

    collapsed = " ".join(chunk.split())
    if collapsed and collapsed != chunk:
        idx = context.find(collapsed, max(0, cursor - 50))
        if idx < 0:
            idx = context.find(collapsed)
        if idx >= 0:
            return idx, idx + len(collapsed)
    return -1, -1


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def count_tokens(text: str) -> int:
    return len(str(text).split())


def truncate_text(text: str, max_chars: int) -> str:
    """Return a single-line preview string without changing metric inputs."""
    value = " ".join(str(text).split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def harmonic(precision: float, recall: float, *, beta: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


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



def first_non_null(values: Any) -> Any:
    """Return the first non-null value from a pandas Series/list-like object."""
    if values is None:
        return None
    try:
        iterable = list(values)
    except TypeError:
        iterable = [values]
    for value in iterable:
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        return value
    return None

def percentile(values: Iterable[Any], p: float) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if arr.empty:
        return 0.0
    return float(np.percentile(arr, p))


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


if __name__ == "__main__":
    main()

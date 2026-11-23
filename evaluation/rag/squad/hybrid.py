#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (
    DEFAULT_HYBRID_WEIGHTS,
    chunks_to_documents,
    evaluate_prefix_retrieval,
    format_weight,
    load_prepared_inputs,
    retrieval_config,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate production hybrid retrieval on prepared SQuAD chunks.")
    parser.add_argument("--input-dir", default="evaluation/rag/results/squad_chunks")
    parser.add_argument("--out", default=None, help="Defaults to <input-dir>/hybrid")
    parser.add_argument("--bm25-k", type=int, required=True, help="BM25 candidate depth inside the hybrid retriever")
    parser.add_argument("--vector-k", type=int, required=True, help="Vector candidate depth inside the hybrid retriever")
    parser.add_argument("--weights", nargs="+", type=float, default=DEFAULT_HYBRID_WEIGHTS, help="BM25 weights to test")
    parser.add_argument(
        "--hybrid-k-values",
        nargs="+",
        type=int,
        default=None,
        help="Final hybrid k values. Defaults to every integer from 1 to N, where N=bm25_k+vector_k.",
    )
    parser.add_argument("--embedding-backend", default="sentence_transformer")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--fail-on-empty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out) if args.out else input_dir / "hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    chunks, queries, qrels_by_query = load_prepared_inputs(input_dir)
    if args.fail_on_empty and (not chunks or not queries or not qrels_by_query):
        raise ValueError("Missing prepared chunks/queries/qrels. Run chunks.py first.")

    n_max = int(args.bm25_k) + int(args.vector_k)
    if args.hybrid_k_values:
        k_values = sorted({int(k) for k in args.hybrid_k_values if 1 <= int(k) <= n_max})
    else:
        k_values = list(range(1, n_max + 1))
    if not k_values:
        raise ValueError("No valid hybrid k values. They must be in [1, bm25_k + vector_k].")

    max_k = max(k_values)
    documents = chunks_to_documents(chunks)

    all_per_query: list[dict] = []
    all_aggregate: list[dict] = []
    build_times: dict[str, float] = {}

    for weight in args.weights:
        run_id = f"hybrid_bm25k{args.bm25_k}_vectork{args.vector_k}_top{max_k}_bm25w{format_weight(weight)}"
        config = retrieval_config(
            strategy="hybrid",
            top_k=max_k,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            bm25_k=args.bm25_k,
            vector_k=args.vector_k,
            bm25_weight=weight,
        )
        per_query, aggregate, build_ms = evaluate_prefix_retrieval(
            retriever_name="hybrid",
            run_id=run_id,
            documents=documents,
            chunks=chunks,
            queries=queries,
            qrels_by_query=qrels_by_query,
            k_values=k_values,
            max_k=max_k,
            config=config,
            out_dir=out_dir,
            extra_columns={
                "bm25_k": int(args.bm25_k),
                "vector_k": int(args.vector_k),
                "bm25_weight": float(weight),
                "vector_weight": 1.0 - float(weight),
            },
        )
        all_per_query.extend(per_query)
        all_aggregate.extend(aggregate)
        build_times[str(weight)] = build_ms

    aggregate_csv = out_dir / "hybrid_by_k.csv"
    descriptive_csv = out_dir / f"hybrid_bm25k{args.bm25_k}_vectork{args.vector_k}_by_k.csv"
    write_csv(aggregate_csv, all_aggregate)
    write_csv(descriptive_csv, all_aggregate)
    write_csv(out_dir / "hybrid_per_query.csv", all_per_query)
    write_json(
        out_dir / "hybrid_run_info.json",
        {
            "retriever": "hybrid",
            "execution": "production hybrid retrieval once at max(hybrid_k_values), then evaluate prefixes",
            "bm25_k": int(args.bm25_k),
            "vector_k": int(args.vector_k),
            "n_max": n_max,
            "hybrid_k_values": k_values,
            "max_k_retrieved": max_k,
            "weights": [float(w) for w in args.weights],
            "num_chunks": len(chunks),
            "num_queries": len(queries),
            "retriever_build_ms_by_weight": build_times,
            "runtime_seconds": time.perf_counter() - start,
            "outputs": {
                "aggregate_csv": str(aggregate_csv),
                "descriptive_csv": str(descriptive_csv),
                "per_query_csv": str(out_dir / "hybrid_per_query.csv"),
            },
        },
    )
    print(json.dumps({"wrote": str(aggregate_csv), "num_queries": len(queries), "k_values": len(k_values)}, indent=2))


if __name__ == "__main__":
    main()

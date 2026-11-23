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
    DEFAULT_COMPONENT_K_VALUES,
    chunks_to_documents,
    evaluate_prefix_retrieval,
    load_prepared_inputs,
    retrieval_config,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate production BM25 retrieval on prepared SQuAD chunks.")
    parser.add_argument("--input-dir", default="evaluation/rag/results/squad_chunks")
    parser.add_argument("--out", default=None, help="Defaults to --input-dir")
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_COMPONENT_K_VALUES)
    parser.add_argument("--embedding-backend", default="sentence_transformer")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--fail-on-empty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out) if args.out else input_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    chunks, queries, qrels_by_query = load_prepared_inputs(input_dir)
    if args.fail_on_empty and (not chunks or not queries or not qrels_by_query):
        raise ValueError("Missing prepared chunks/queries/qrels. Run chunks.py first.")

    k_values = sorted({int(k) for k in args.k_values})
    max_k = max(k_values)
    documents = chunks_to_documents(chunks)
    config = retrieval_config(
        strategy="bm25",
        top_k=max_k,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
    )

    per_query, aggregate, build_ms = evaluate_prefix_retrieval(
        retriever_name="bm25",
        run_id=f"bm25_top{max_k}",
        documents=documents,
        chunks=chunks,
        queries=queries,
        qrels_by_query=qrels_by_query,
        k_values=k_values,
        max_k=max_k,
        config=config,
        out_dir=out_dir,
    )

    write_csv(out_dir / "bm25_by_k.csv", aggregate)
    write_csv(out_dir / "bm25_per_query.csv", per_query)
    write_json(
        out_dir / "bm25_run_info.json",
        {
            "retriever": "bm25",
            "execution": "retrieve once at max(k_values), then evaluate prefixes",
            "k_values": k_values,
            "max_k_retrieved": max_k,
            "num_chunks": len(chunks),
            "num_queries": len(queries),
            "retriever_build_ms": build_ms,
            "runtime_seconds": time.perf_counter() - start,
            "outputs": {
                "aggregate_csv": str(out_dir / "bm25_by_k.csv"),
                "per_query_csv": str(out_dir / "bm25_per_query.csv"),
            },
        },
    )
    print(json.dumps({"wrote": str(out_dir / "bm25_by_k.csv"), "num_queries": len(queries)}, indent=2))


if __name__ == "__main__":
    main()

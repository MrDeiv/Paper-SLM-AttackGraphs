#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (
    ChunkRecord,
    ContextRecord,
    QrelRecord,
    QueryRecord,
    count_tokens,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SQuAD chunks and qrels for retrieval evaluation.")
    parser.add_argument("--dataset", default="rajpurkar/squad")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="evaluation/rag/results/squad_chunks")
    parser.add_argument("--limit-contexts", type=int, default=None)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--chunking-strategy", default="semantic")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--fail-on-empty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    ds = load_dataset(args.dataset, split=args.split)
    contexts, queries = build_contexts_and_queries(
        ds,
        limit_contexts=args.limit_contexts,
        limit_questions=args.limit_questions,
    )

    chunks, chunker_backend = chunk_contexts_with_pipeline(
        contexts,
        chunking_strategy=args.chunking_strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    qrels = build_qrels(queries, chunks)
    query_ids_with_qrels = {q.query_id for q in qrels}
    queries = [q for q in queries if q.query_id in query_ids_with_qrels]

    if args.fail_on_empty:
        if not contexts:
            raise ValueError("No SQuAD contexts loaded")
        if not chunks:
            raise ValueError("No chunks generated")
        if not queries:
            raise ValueError("No questions with answer-bearing chunks")
        if not qrels:
            raise ValueError("No qrels generated")

    write_csv(out_dir / "contexts.csv", [asdict(ctx) for ctx in contexts])
    write_csv(out_dir / "queries.csv", [asdict(query) for query in queries])
    write_csv(out_dir / "qrels.csv", [asdict(qrel) for qrel in qrels])
    write_jsonl(out_dir / "chunks.jsonl", [asdict(chunk) for chunk in chunks])

    log = {
        "dataset": args.dataset,
        "split": args.split,
        "chunker_backend": chunker_backend,
        "chunking_strategy": args.chunking_strategy,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "num_original_contexts": len(contexts),
        "num_generated_chunks": len(chunks),
        "num_questions_with_qrels": len(queries),
        "num_qrels": len(qrels),
        "runtime_seconds": time.perf_counter() - start,
        "outputs": {
            "contexts": str(out_dir / "contexts.csv"),
            "queries": str(out_dir / "queries.csv"),
            "chunks": str(out_dir / "chunks.jsonl"),
            "qrels": str(out_dir / "qrels.csv"),
        },
    }
    write_json(out_dir / "chunk_log.json", log)

    print(json.dumps(log, indent=2))


def build_contexts_and_queries(
    ds: Dataset,
    *,
    limit_contexts: int | None,
    limit_questions: int | None,
) -> tuple[list[ContextRecord], list[QueryRecord]]:
    """Keep one source context per unique paragraph and many questions per context."""
    contexts_by_text: dict[str, ContextRecord] = {}
    queries: list[QueryRecord] = []

    for idx, item in enumerate(tqdm(ds, desc="Loading SQuAD")):
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
            contexts_by_text[context] = ContextRecord(
                context_id=context_id,
                title=title,
                text=context,
                source_index=idx,
            )

        ctx = contexts_by_text.get(context)
        if ctx is None:
            continue

        answer_text = str(answer_texts[0])
        answer_start = int(answer_starts[0])
        answer_end = answer_start + len(answer_text)
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


def chunk_contexts_with_pipeline(
    contexts: list[ContextRecord],
    *,
    chunking_strategy: str,
    chunk_size: int,
    overlap: int,
) -> tuple[list[ChunkRecord], str]:
    """Chunk contexts using the pipeline DocumentFactory when available.

    Character offsets are recovered by matching chunk text back to the original
    SQuAD paragraph. Those offsets are used to identify answer-bearing chunks.
    """
    try:
        from src.DocumentFactory import DocumentFactory

        chunks: list[ChunkRecord] = []
        for ctx in tqdm(contexts, desc="Chunking with DocumentFactory"):
            docs = call_document_factory(
                DocumentFactory,
                text=ctx.text,
                document_id=ctx.context_id,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            cursor = 0
            for idx, doc in enumerate(docs, start=1):
                text = get_doc_text(doc).strip()
                if not text:
                    continue
                metadata = get_doc_metadata(doc)
                start_char = int(metadata.get("start_char", metadata.get("start", -1)) or -1)
                end_char = int(metadata.get("end_char", metadata.get("end", -1)) or -1)
                if start_char < 0 or end_char < 0:
                    start_char, end_char = locate_chunk(ctx.text, text, cursor)
                if start_char >= 0:
                    cursor = max(cursor, end_char)
                chunks.append(
                    ChunkRecord(
                        chunk_id=str(metadata.get("chunk_id") or f"{ctx.context_id}_c{idx:05d}"),
                        context_id=ctx.context_id,
                        title=ctx.title,
                        chunk_index=idx,
                        text=text,
                        start_char=start_char,
                        end_char=end_char,
                        token_count=count_tokens(text),
                    )
                )
        return chunks, "src.DocumentFactory.DocumentFactory"
    except Exception as exc:
        print(f"[WARN] DocumentFactory unavailable or incompatible ({exc!r}); falling back to ClusterSemanticChunker.")
        return chunk_contexts_fallback(contexts, max_chunk_size=chunk_size), "ClusterSemanticChunker fallback"


def call_document_factory(
    DocumentFactory: Any,
    *,
    text: str,
    document_id: str,
    chunking_strategy: str,
    chunk_size: int,
    overlap: int,
) -> list[Any]:
    config = {"max_chunk_tokens": chunk_size, "overlap_tokens": overlap}
    # Different local versions of DocumentFactory may expose from_text as a
    # static method or instance method. These attempts keep the script robust.
    attempts = [
        lambda: DocumentFactory.from_text(
            text=text,
            document_id=document_id,
            case_id="squad",
            chunking_strategy=chunking_strategy,
            chunking_config=config,
        ),
        lambda: DocumentFactory.from_text(
            text,
            document_id=document_id,
            case_id="squad",
            chunking_strategy=chunking_strategy,
            chunking_config=config,
        ),
        lambda: DocumentFactory().from_text(
            text=text,
            document_id=document_id,
            case_id="squad",
            chunking_strategy=chunking_strategy,
            chunking_config=config,
        ),
        lambda: DocumentFactory().from_text(text, document_id=document_id),
    ]
    errors: list[str] = []
    for attempt in attempts:
        try:
            docs = attempt()
            return list(docs)
        except Exception as exc:
            errors.append(repr(exc))
    raise RuntimeError("Could not call DocumentFactory.from_text: " + "; ".join(errors))


def chunk_contexts_fallback(contexts: list[ContextRecord], *, max_chunk_size: int) -> list[ChunkRecord]:
    from chromadb.utils import embedding_functions
    from chunking_evaluation.chunking import ClusterSemanticChunker

    default_ef = embedding_functions.DefaultEmbeddingFunction()
    chunker = ClusterSemanticChunker(default_ef, max_chunk_size=max_chunk_size)
    chunks: list[ChunkRecord] = []
    for ctx in tqdm(contexts, desc="Chunking with fallback"):
        cursor = 0
        for idx, raw_chunk in enumerate(chunker.split_text(ctx.text), start=1):
            text = str(raw_chunk).strip()
            if not text:
                continue
            start_char, end_char = locate_chunk(ctx.text, text, cursor)
            if start_char >= 0:
                cursor = max(cursor, end_char)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{ctx.context_id}_c{idx:05d}",
                    context_id=ctx.context_id,
                    title=ctx.title,
                    chunk_index=idx,
                    text=text,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=count_tokens(text),
                )
            )
    return chunks


def build_qrels(queries: list[QueryRecord], chunks: list[ChunkRecord]) -> list[QrelRecord]:
    """A chunk is relevant when it overlaps the SQuAD gold answer span."""
    chunks_by_context: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        chunks_by_context.setdefault(chunk.context_id, []).append(chunk)

    qrels: list[QrelRecord] = []
    seen: set[tuple[str, str]] = set()
    for query in queries:
        for chunk in chunks_by_context.get(query.context_id, []):
            match_type = ""
            if chunk.start_char >= 0 and chunk.end_char >= 0 and overlaps(
                chunk.start_char,
                chunk.end_char,
                query.answer_start,
                query.answer_end,
            ):
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


def get_doc_text(doc: Any) -> str:
    if hasattr(doc, "page_content"):
        return str(doc.page_content)
    if isinstance(doc, dict):
        return str(doc.get("page_content") or doc.get("text") or "")
    return str(doc)


def get_doc_metadata(doc: Any) -> dict[str, Any]:
    if hasattr(doc, "metadata"):
        return dict(doc.metadata or {})
    if isinstance(doc, dict):
        return dict(doc.get("metadata") or {})
    return {}


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
    if idx >= 0:
        return idx, idx + len(collapsed)
    return -1, -1


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def make_context_id(*, title: str, text: str, index: int) -> str:
    return f"squad_{safe_id(title)}_{index:06d}_{stable_hash(text)[:10]}"


def safe_id(text: str) -> str:
    value = "".join(ch if ch.isalnum() else "_" for ch in str(text).lower())
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_")[:60] or "context"


def stable_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", errors="ignore")).hexdigest()


if __name__ == "__main__":
    main()

"""
Retrieval abstraction for CTI enrichment experiments.

This module provides a common interface for BM25-only, vector-only, hybrid,
and no-retrieval variants. It returns standardized RetrievalRecord objects so
that every downstream stage can preserve chunk provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.schemas import RetrievalRecord, RetrievedChunkRecord
from src.stores.DocumentStore import DocumentStore


@dataclass
class RetrieverBundle:
    name: str
    retriever: Any | None
    top_k: int
    chunking_strategy: str

    def invoke(self, query: str) -> list[Document]:
        if self.retriever is None:
            return []
        if hasattr(self.retriever, "invoke"):
            return list(self.retriever.invoke(query))
        if hasattr(self.retriever, "get_relevant_documents"):
            return list(self.retriever.get_relevant_documents(query))
        raise TypeError(f"Unsupported retriever object: {type(self.retriever)!r}")


def build_retriever(documents: list[Document], config: dict[str, Any]) -> RetrieverBundle:
    """
    Build a retriever according to an experiment config.

    Supported retrieval.strategy values:
    - none
    - bm25
    - vector
    - hybrid
    """
    retrieval_cfg = config.get("retrieval", {})
    chunking_cfg = config.get("chunking", {})
    models_cfg = config.get("models", {})

    enabled = bool(retrieval_cfg.get("enabled", True))
    strategy = str(retrieval_cfg.get("strategy", "hybrid")).lower()
    top_k = int(retrieval_cfg.get("top_k", retrieval_cfg.get("k", 10)))
    chunking_strategy = str(chunking_cfg.get("strategy", "unknown"))

    if not enabled or strategy in {"none", "no_rag", "slm_only"}:
        return RetrieverBundle(
            name="none",
            retriever=None,
            top_k=0,
            chunking_strategy=chunking_strategy,
        )

    if not documents:
        raise ValueError("Cannot build a retriever with an empty document list")

    if strategy == "bm25":
        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = top_k
        return RetrieverBundle(
            name="bm25",
            retriever=bm25,
            top_k=top_k,
            chunking_strategy=chunking_strategy,
        )

    if strategy == "vector":
        embedding_backend = str(models_cfg.get("embedding_backend", "sentence_transformer"))
        embedding_model = _embedding_model_from_config(models_cfg, embedding_backend)

        store = DocumentStore(
            k=top_k,
            embedding_model=embedding_model,
            embedding_backend=embedding_backend,
        )
        store.ingest(documents)

        return RetrieverBundle(
            name="vector",
            retriever=store.retriever,
            top_k=top_k,
            chunking_strategy=chunking_strategy,
        )

    if strategy == "hybrid":
        bm25_k = int(retrieval_cfg.get("bm25_k", top_k))
        vector_k = int(retrieval_cfg.get("vector_k", top_k))
        alpha = float(retrieval_cfg.get("hybrid_alpha", 0.5))
        alpha = min(max(alpha, 0.0), 1.0)

        bm25 = BM25Retriever.from_documents(documents)
        bm25.k = bm25_k

        embedding_backend = str(models_cfg.get("embedding_backend", "sentence_transformer"))
        embedding_model = _embedding_model_from_config(models_cfg, embedding_backend)

        store = DocumentStore(
            k=vector_k,
            embedding_model=embedding_model,
            embedding_backend=embedding_backend,
        )
        store.ingest(documents)

        ensemble = EnsembleRetriever(
            retrievers=[bm25, store.retriever],
            weights=[alpha, 1.0 - alpha],
        )

        return RetrieverBundle(
            name="hybrid",
            retriever=ensemble,
            top_k=top_k,
            chunking_strategy=chunking_strategy,
        )

    raise ValueError(f"Unsupported retrieval.strategy: {strategy!r}")


def retrieve_with_provenance(
    bundle: RetrieverBundle,
    *,
    query: str,
    run_id: str,
    case_id: str,
    attack_step_id: str,
) -> RetrievalRecord:
    """
    Retrieve chunks and convert them into a standardized RetrievalRecord.
    """
    docs = bundle.invoke(query)
    unique_docs = _deduplicate_documents(docs)

    if bundle.top_k:
        unique_docs = unique_docs[: bundle.top_k]

    results: list[RetrievedChunkRecord] = []

    for rank, doc in enumerate(unique_docs, start=1):
        metadata = dict(doc.metadata or {})

        results.append(
            RetrievedChunkRecord(
                rank=rank,
                chunk_id=str(metadata.get("chunk_id", f"missing_chunk_id_{rank:04d}")),
                document_id=str(metadata.get("document_id", "missing_document_id")),
                source_path=str(metadata.get("source_path", metadata.get("source", ""))),
                text=doc.page_content,
                score=_optional_float(metadata.get("score")),
                bm25_score=_optional_float(metadata.get("bm25_score")),
                vector_score=_optional_float(metadata.get("vector_score")),
                metadata=metadata,
            )
        )

    return RetrievalRecord(
        run_id=run_id,
        case_id=case_id,
        attack_step_id=attack_step_id,
        retriever=bundle.name,
        chunking_strategy=bundle.chunking_strategy,
        query=query,
        top_k=bundle.top_k,
        results=results,
    )


def _embedding_model_from_config(models_cfg: dict[str, Any], embedding_backend: str) -> str:
    """
    Resolve the embedding model name from config.

    For sentence-transformers, prefer models.sentence_transformer.
    For Ollama, prefer models.embeddings.
    """
    embedding_backend = embedding_backend.strip().lower()

    if embedding_backend in {"sentence_transformer", "sentence-transformer", "local_sentence_transformer"}:
        return str(models_cfg.get("sentence_transformer", models_cfg.get("embeddings", "all-MiniLM-L6-v2")))

    if embedding_backend == "ollama":
        return str(models_cfg.get("embeddings", "embeddinggemma"))

    return str(models_cfg.get("embeddings", "all-MiniLM-L6-v2"))


def _deduplicate_documents(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    unique: list[Document] = []

    for idx, doc in enumerate(docs):
        metadata = dict(doc.metadata or {})
        key = str(metadata.get("chunk_id") or metadata.get("source") or idx)

        if key in seen:
            continue

        seen.add(key)
        unique.append(doc)

    return unique


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None
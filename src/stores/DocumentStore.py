"""
Vector document store
"""

from __future__ import annotations

from typing import Iterable, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


class SentenceTransformerEmbeddings(Embeddings):
    """
    Minimal LangChain-compatible wrapper around sentence-transformers.

    This keeps vector retrieval local-first without requiring Ollama's embedding
    endpoint. The model is loaded once and then used for document/query vectors.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise ImportError(
                "sentence-transformers is required for embedding_backend='sentence_transformer'. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


class DocumentStore:
    """
    Thin wrapper around a FAISS vector store.

    Parameters
    ----------
    k:
        Number of documents returned by the retriever.
    embedding_model:
        Embedding model name.
    embedding_backend:
        Either 'sentence_transformer' or 'ollama'.
    """

    def __init__(
        self,
        k: int = 3,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_backend: str = "sentence_transformer",
    ):
        self.k = k
        self.embedding_model = embedding_model
        self.embedding_backend = embedding_backend
        self.embeddings = self._build_embeddings(
            backend=embedding_backend,
            model_name=embedding_model,
        )
        self.vector_db: FAISS | None = None
        self.retriever = None

    def ingest(self, documents: list[Document]) -> None:
        """
        Ingest documents into FAISS and expose a LangChain retriever.
        """
        if not documents:
            raise ValueError("DocumentStore.ingest received an empty document list")

        self.vector_db = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )

    def add_documents(self, documents: Iterable[Document]) -> list[str]:
        """
        Add documents to an already-created FAISS store.
        """
        docs = list(documents)

        if self.vector_db is None:
            self.ingest(docs)
            return []

        return self.vector_db.add_documents(docs)

    @staticmethod
    def _build_embeddings(backend: str, model_name: str) -> Embeddings:
        backend = backend.strip().lower()

        if backend in {"sentence_transformer", "sentence-transformer", "local_sentence_transformer"}:
            return SentenceTransformerEmbeddings(model_name=model_name)

        if backend == "ollama":
            try:
                from langchain_ollama import OllamaEmbeddings
            except Exception as exc:
                raise ImportError(
                    "langchain-ollama is required for embedding_backend='ollama'. "
                    "Install it with: pip install langchain-ollama"
                ) from exc

            return OllamaEmbeddings(model=model_name)

        raise ValueError(
            f"Unsupported embedding backend: {backend!r}. "
            "Allowed values: 'sentence_transformer', 'ollama'."
        )
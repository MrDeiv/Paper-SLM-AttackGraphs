"""
Document loading and chunking utilities.

This version keeps the legacy DocumentFactory.from_file(file) interface used by
app.py, while adding stable provenance metadata required by the revised
experiment pipeline:

- case_id
- document_id
- chunk_id
- chunk_index
- source path
- chunking_strategy

The default chunking strategy is semantic when chunking_evaluation is available;
otherwise it falls back to fixed-size recursive character chunking. Experiment
scripts should pass chunking_strategy explicitly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import PyPDF2
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter

from src.CustomCSVLoader import CustomCSVLoader
from src.CustomJSONLoader import CustomJSONLoader
from src.CustomYMLLoader import CustomYMLLoader
from src.id_utils import make_chunk_id, make_document_id


DEFAULT_MAX_CHUNK_TOKENS = 200
DEFAULT_OVERLAP_TOKENS = 0
SUPPORTED_EXTENSIONS = {".csv", ".json", ".yml", ".yaml", ".txt", ".pdf", ".html", ".htm", ".xml"}


class DocumentFactory:
    """
    Factory for converting heterogeneous CTI files into LangChain Documents.
    """

    @staticmethod
    def from_text(
        text: str,
        *,
        case_id: str | None = None,
        document_id: str | None = None,
        source: str = "text",
        chunking_strategy: str = "semantic",
        chunking_config: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        document_id = document_id or make_document_id(source, case_id=case_id)
        chunks = _split_text(text, chunking_strategy=chunking_strategy, chunking_config=chunking_config)
        return _documents_from_chunks(
            chunks,
            case_id=case_id,
            document_id=document_id,
            source=source,
            chunking_strategy=chunking_strategy,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def from_text_file(
        file: str,
        *,
        case_id: str | None = None,
        document_id: str | None = None,
        chunking_strategy: str = "semantic",
        chunking_config: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        _assert_exists(file)
        text = Path(file).read_text(encoding="utf-8", errors="ignore")
        return DocumentFactory.from_text(
            text,
            case_id=case_id,
            document_id=document_id,
            source=file,
            chunking_strategy=chunking_strategy,
            chunking_config=chunking_config,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def from_pdf(
        file: str,
        *,
        case_id: str | None = None,
        document_id: str | None = None,
        chunking_strategy: str = "semantic",
        chunking_config: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        _assert_exists(file)
        reader = PyPDF2.PdfReader(file)
        pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
        text = "\n\n".join(pages)
        return DocumentFactory.from_text(
            text,
            case_id=case_id,
            document_id=document_id,
            source=file,
            chunking_strategy=chunking_strategy,
            chunking_config=chunking_config,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def from_json(
        file: str,
        *,
        case_id: str | None = None,
        document_id: str | None = None,
        chunking_strategy: str = "semantic",
        chunking_config: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        _assert_exists(file)
        document_id = document_id or make_document_id(file, case_id=case_id)
        loaded_json = CustomJSONLoader(file).load()

        # Preserve the original JSON-aware splitter, then annotate its outputs.
        split_docs = RecursiveJsonSplitter().create_documents([loaded_json], metadatas=[{"source": file}])
        return _annotate_documents(
            split_docs,
            case_id=case_id,
            document_id=document_id,
            source=file,
            chunking_strategy="json_recursive",
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def from_csv(
        file: str,
        *,
        case_id: str | None = None,
        document_id: str | None = None,
        chunking_strategy: str = "row",
        chunking_config: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        _assert_exists(file)
        document_id = document_id or make_document_id(file, case_id=case_id)
        rows = CustomCSVLoader(file).load()
        docs = [Document(page_content=str(row), metadata={"source": file}) for row in rows]
        return _annotate_documents(
            docs,
            case_id=case_id,
            document_id=document_id,
            source=file,
            chunking_strategy=chunking_strategy,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def from_yml(
        file: str,
        *,
        case_id: str | None = None,
        document_id: str | None = None,
        chunking_strategy: str = "yaml_recursive",
        chunking_config: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        _assert_exists(file)
        document_id = document_id or make_document_id(file, case_id=case_id)
        loaded_yml = CustomYMLLoader(file).load()
        split_docs = RecursiveJsonSplitter().create_documents([loaded_yml], metadatas=[{"source": file}])
        return _annotate_documents(
            split_docs,
            case_id=case_id,
            document_id=document_id,
            source=file,
            chunking_strategy=chunking_strategy,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def from_html(
        file: str,
        *,
        case_id: str | None = None,
        document_id: str | None = None,
        chunking_strategy: str = "semantic",
        chunking_config: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        _assert_exists(file)
        html = Path(file).read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text("\n", strip=True)
        return DocumentFactory.from_text(
            text,
            case_id=case_id,
            document_id=document_id,
            source=file,
            chunking_strategy=chunking_strategy,
            chunking_config=chunking_config,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def from_xml(
        file: str,
        *,
        case_id: str | None = None,
        document_id: str | None = None,
        chunking_strategy: str = "semantic",
        chunking_config: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        _assert_exists(file)
        xml = Path(file).read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(xml, "xml")
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text("\n", strip=True)
        return DocumentFactory.from_text(
            text,
            case_id=case_id,
            document_id=document_id,
            source=file,
            chunking_strategy=chunking_strategy,
            chunking_config=chunking_config,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def from_file(
        file: str,
        *,
        case_id: str | None = None,
        document_id: str | None = None,
        chunking_strategy: str = "semantic",
        chunking_config: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Process one file into LangChain Documents with provenance metadata.

        Parameters are keyword-only after `file` to preserve compatibility with
        the legacy call DocumentFactory.from_file(file).
        """
        _assert_exists(file)
        ext = Path(file).suffix.lower()

        processors = {
            ".csv": DocumentFactory.from_csv,
            ".json": DocumentFactory.from_json,
            ".yml": DocumentFactory.from_yml,
            ".yaml": DocumentFactory.from_yml,
            ".txt": DocumentFactory.from_text_file,
            ".pdf": DocumentFactory.from_pdf,
            ".html": DocumentFactory.from_html,
            ".htm": DocumentFactory.from_html,
            ".xml": DocumentFactory.from_xml,
        }

        processor = processors.get(ext, DocumentFactory.from_text_file)
        return processor(
            file,
            case_id=case_id,
            document_id=document_id,
            chunking_strategy=chunking_strategy,
            chunking_config=chunking_config,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def from_directory(
        directory: str,
        *,
        case_id: str | None = None,
        chunking_strategy: str = "semantic",
        chunking_config: dict[str, Any] | None = None,
        recursive: bool = True,
    ) -> list[Document]:
        """
        Load all supported files from a directory.
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        pattern = "**/*" if recursive else "*"
        documents: list[Document] = []
        for path in sorted(directory_path.glob(pattern)):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".zip":
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            documents.extend(
                DocumentFactory.from_file(
                    str(path),
                    case_id=case_id,
                    chunking_strategy=chunking_strategy,
                    chunking_config=chunking_config,
                )
            )
        return documents


def _assert_exists(file: str) -> None:
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} not found")


def _split_text(
    text: str,
    *,
    chunking_strategy: str,
    chunking_config: dict[str, Any] | None = None,
) -> list[str]:
    chunking_config = chunking_config or {}
    strategy = chunking_strategy.lower()

    if strategy == "semantic":
        semantic_chunks = _semantic_split(text, chunking_config)
        if semantic_chunks:
            return semantic_chunks

    return _fixed_split(text, chunking_config)


def _semantic_split(text: str, chunking_config: dict[str, Any]) -> list[str]:
    """
    Try semantic chunking; return an empty list if unavailable.
    """
    try:
        from chromadb.utils import embedding_functions
        from chunking_evaluation.chunking import ClusterSemanticChunker

        max_chunk_size = int(chunking_config.get("max_chunk_tokens", chunking_config.get("chunk_size", DEFAULT_MAX_CHUNK_TOKENS)))
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        chunker = ClusterSemanticChunker(default_ef, max_chunk_size=max_chunk_size)
        chunks = [chunk.strip() for chunk in chunker.split_text(text) if chunk.strip()]
        return chunks
    except Exception:
        return []


def _fixed_split(text: str, chunking_config: dict[str, Any]) -> list[str]:
    chunk_size = int(chunking_config.get("max_chunk_tokens", chunking_config.get("chunk_size", DEFAULT_MAX_CHUNK_TOKENS)))
    chunk_overlap = int(chunking_config.get("overlap_tokens", chunking_config.get("chunk_overlap", DEFAULT_OVERLAP_TOKENS)))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def _documents_from_chunks(
    chunks: Iterable[str],
    *,
    case_id: str | None,
    document_id: str,
    source: str,
    chunking_strategy: str,
    extra_metadata: dict[str, Any] | None,
) -> list[Document]:
    docs = [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]
    return _annotate_documents(
        docs,
        case_id=case_id,
        document_id=document_id,
        source=source,
        chunking_strategy=chunking_strategy,
        extra_metadata=extra_metadata,
    )


def _annotate_documents(
    docs: list[Document],
    *,
    case_id: str | None,
    document_id: str,
    source: str,
    chunking_strategy: str,
    extra_metadata: dict[str, Any] | None,
) -> list[Document]:
    annotated: list[Document] = []
    source_path = str(source)
    source_name = Path(source_path).name
    source_type = Path(source_path).suffix.lower().lstrip(".") or "text"
    metadata_base = dict(extra_metadata or {})

    for i, doc in enumerate(docs):
        metadata = dict(doc.metadata or {})
        metadata.update(metadata_base)
        metadata.update(
            {
                "source": source_path,
                "source_path": source_path,
                "source_name": source_name,
                "source_type": source_type,
                "case_id": case_id or "unknown_case",
                "document_id": document_id,
                "chunk_id": make_chunk_id(document_id, i),
                "chunk_index": i,
                "chunking_strategy": chunking_strategy,
                "token_count": _rough_token_count(doc.page_content),
            }
        )
        annotated.append(Document(page_content=doc.page_content, metadata=metadata))

    return annotated


def _rough_token_count(text: str) -> int:
    """
    Lightweight token proxy used for logging and annotation exports.
    """
    return len(text.split())
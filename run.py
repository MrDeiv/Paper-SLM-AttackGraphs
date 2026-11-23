from __future__ import annotations

import json
import logging
import logging.config
import os
import re
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import dotenv
from bs4 import BeautifulSoup
from markdown import markdown
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:
    SentenceTransformer = None
    st_util = None

from langchain.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

from src.DocumentFactory import DocumentFactory
from src.STIXParser import STIXParser
from src.group_attack_patterns import group_attack_patterns
from src.stores.DocumentStore import DocumentStore


CONFIG_FILE = Path("configs/config.json")
SAFE_DELIMITER = " ||| "


def remove_markdown(text: str) -> str:
    """Convert a markdown/plain-text model response to plain text."""
    text = str(text or "")
    mk = markdown(text)
    warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
    return "".join(BeautifulSoup(mk, features="html.parser").find_all(text=True)).strip()


def squash_ws(text: Any) -> str:
    """Normalize whitespace while preserving the text content."""
    return " ".join(str(text or "").split())


def token_count(text: Any) -> int:
    """Deterministic whitespace token count used only for evaluation metrics."""
    return len(str(text or "").split())


def slugify(value: Any) -> str:
    """Create stable, readable identifiers from STIX names/tactics."""
    text = str(value or "unknown").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@dataclass
class TimedResult:
    value: Any
    latency_ms: float


def timed_call(fn, *args, **kwargs) -> TimedResult:
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return TimedResult(value=value, latency_ms=(time.perf_counter() - start) * 1000.0)


class JsonlLogger:
    """Append-only JSONL execution log.

    The log is intentionally verbose: it records stage boundaries, retrieval
    queries, retrieved evidence IDs, generation outputs, errors, and latencies.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("w", encoding="utf-8")

    def write(self, event: str, **payload: Any) -> None:
        row = {"ts": now_iso(), "event": event, **payload}
        self.fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required = ["CASE_ID", "STIX_FILE", "DOCUMENTS_DIR", "OUTPUT_DIR", "MODELS"]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing required configuration keys in {CONFIG_FILE}: {missing}")

    if "TEXT_GENERATION" not in config["MODELS"]:
        raise KeyError("Missing MODELS.TEXT_GENERATION in configs/config.json")

    return config


def cfg_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read nested config values using dot notation."""
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def get_rag_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the evaluated hybrid RAG configuration.

    Primary source is the RAG block in configs/config.json. Legacy top-level keys
    are accepted only for backward compatibility with the original script.
    """
    rag = config.get("RAG", {}) if isinstance(config.get("RAG"), dict) else {}

    bm25_k = int(rag.get("bm25_k", config.get("BM25_k")))
    vector_k = int(rag.get("vector_k", config.get("k")))
    top_k = int(rag.get("top_k", config.get("TOP_K", max(bm25_k, vector_k))))
    hybrid_alpha = float(rag.get("hybrid_alpha", config.get("HYBRID_ALPHA", 0.25)))

    return {
        "strategy": "hybrid",
        "bm25_k": bm25_k,
        "vector_k": vector_k,
        "top_k": top_k,
        "hybrid_alpha": hybrid_alpha,
        "vector_weight": 1.0 - hybrid_alpha,
    }


def output_dir_for_case(config: dict[str, Any]) -> Path:
    return Path(config["OUTPUT_DIR"]) / str(config["CASE_ID"])


def configure_logging(case_output_dir: Path) -> logging.Logger:
    case_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = case_output_dir / "app.log"

    logging.config.dictConfig({"version": 1, "disable_existing_loggers": False})
    logging.basicConfig(
        level=logging.INFO,
        filename=str(log_file),
        filemode="w",
        format="[%(asctime)s %(levelname)s] %(message)s",
    )
    logger = logging.getLogger("cti_enrichment")
    logger.info("Logging initialized at %s", log_file)
    return logger


def load_documents(config: dict[str, Any], logger: logging.Logger, exec_log: JsonlLogger) -> list[Document]:
    """Load and chunk documents using the existing DocumentFactory component."""
    documents_dir = Path(config["DOCUMENTS_DIR"])
    if not documents_dir.exists():
        raise FileNotFoundError(f"DOCUMENTS_DIR does not exist: {documents_dir}")

    # Original script loaded DOCUMENTS_DIR/other. This version expects
    # DOCUMENTS_DIR to point to the exact directory containing the case documents;
    # optional DOCUMENTS_SUBDIR is still supported for compatibility.
    subdir = config.get("DOCUMENTS_SUBDIR")
    if subdir:
        documents_dir = documents_dir / str(subdir)

    files = [p for p in documents_dir.rglob("*") if p.is_file() and not p.name.endswith(".zip")]
    documents: list[Document] = []

    exec_log.write("document_loading_started", documents_dir=str(documents_dir), files=len(files))
    for file_path in tqdm(files, desc="Loading/chunking documents"):
        try:
            loaded = DocumentFactory.from_file(str(file_path))
            for doc in loaded:
                ensure_document_metadata(doc, config, file_path)
                documents.append(doc)
            exec_log.write("document_loaded", path=str(file_path), chunks=len(loaded))
        except Exception as exc:
            logger.exception("Failed loading document %s", file_path)
            exec_log.write(
                "document_loading_error",
                path=str(file_path),
                error=str(exc),
                traceback=traceback.format_exc(),
            )

    assign_stable_chunk_ids(documents, config["CASE_ID"])
    exec_log.write("document_loading_finished", chunks=len(documents))
    logger.info("Loaded %d evidence chunks", len(documents))
    return documents


def ensure_document_metadata(doc: Document, config: dict[str, Any], file_path: Path) -> None:
    metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
    metadata.setdefault("case_id", config["CASE_ID"])
    metadata.setdefault("source_path", str(file_path))
    metadata.setdefault("source", metadata.get("source") or file_path.name)
    metadata.setdefault("title", metadata.get("title") or file_path.stem)
    metadata.setdefault("document_id", metadata.get("document_id") or slugify(file_path.stem))
    metadata.setdefault("token_count", token_count(doc.page_content))
    doc.metadata = metadata


def assign_stable_chunk_ids(documents: list[Document], case_id: str) -> None:
    """Ensure every chunk has a stable deterministic chunk_id."""
    per_doc_counter: dict[str, int] = {}
    for idx, doc in enumerate(documents):
        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        document_id = str(metadata.get("document_id") or f"doc{idx:04d}")
        per_doc_counter[document_id] = per_doc_counter.get(document_id, 0) + 1
        chunk_index = metadata.get("chunk_index", per_doc_counter[document_id])

        metadata["chunk_index"] = int(chunk_index) if str(chunk_index).isdigit() else chunk_index
        metadata["chunk_id"] = str(
            metadata.get("chunk_id")
            or f"{case_id}-{slugify(document_id)}-chunk{int(per_doc_counter[document_id]):04d}"
        )
        metadata["token_count"] = token_count(doc.page_content)
        doc.metadata = metadata


def write_chunks_jsonl(path: Path, documents: list[Document]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            row = {
                **(doc.metadata or {}),
                "text": doc.page_content,
            }
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def build_evaluated_hybrid_retriever(
    documents: list[Document],
    rag_config: dict[str, Any],
    logger: logging.Logger,
    exec_log: JsonlLogger,
) -> EnsembleRetriever:
    """Build the same hybrid RAG family evaluated before.

    Evaluated setting:
    - BM25 candidate depth = rag_config["bm25_k"]
    - vector candidate depth = rag_config["vector_k"]
    - final returned chunks = rag_config["top_k"] after fusion
    - BM25 weight = rag_config["hybrid_alpha"]
    - vector weight = 1 - hybrid_alpha
    """
    exec_log.write("rag_build_started", rag_config=rag_config, chunks=len(documents))
    start = time.perf_counter()

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = rag_config["bm25_k"]

    docstore = DocumentStore(k=rag_config["vector_k"])
    docstore.ingest(documents)

    # Some DocumentStore retrievers expose search_kwargs. Set it when possible
    # without assuming a specific vector-store implementation.
    if hasattr(docstore.retriever, "search_kwargs") and isinstance(docstore.retriever.search_kwargs, dict):
        docstore.retriever.search_kwargs["k"] = rag_config["vector_k"]

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, docstore.retriever],
        weights=[rag_config["hybrid_alpha"], rag_config["vector_weight"]],
    )

    build_ms = (time.perf_counter() - start) * 1000.0
    logger.info("Hybrid retriever built in %.2f ms", build_ms)
    exec_log.write("rag_build_finished", build_ms=build_ms)
    return retriever


def retrieve_evidence(
    retriever: EnsembleRetriever,
    query: str,
    rag_config: dict[str, Any],
    exec_log: JsonlLogger,
    event_context: dict[str, Any],
) -> TimedResult:
    """Retrieve evidence and truncate only the final fused list to top_k."""
    result = timed_call(retriever.invoke, query)
    docs = list(result.value or [])[: rag_config["top_k"]]
    payload = {
        **event_context,
        "query": query,
        "latency_ms": result.latency_ms,
        "retrieved_chunks": len(docs),
        "evidence_chunk_ids": [chunk_id(doc) for doc in docs],
    }
    exec_log.write("retrieval_finished", **payload)
    return TimedResult(value=docs, latency_ms=result.latency_ms)


def chunk_id(doc: Document) -> str:
    return str((doc.metadata or {}).get("chunk_id", ""))


def evidence_payload(docs: list[Document]) -> dict[str, Any]:
    """Return deterministic evidence metadata attached to generated fields."""
    return {
        "evidence_chunk_ids": [chunk_id(doc) for doc in docs],
        "evidence_document_ids": [str((doc.metadata or {}).get("document_id", "")) for doc in docs],
        "evidence_sources": [str((doc.metadata or {}).get("source", "")) for doc in docs],
        "evidence_titles": [str((doc.metadata or {}).get("title", "")) for doc in docs],
        "evidence_texts": [doc.page_content for doc in docs],
        "evidence_ranks": list(range(1, len(docs) + 1)),
        "evidence_token_count": sum(token_count(doc.page_content) for doc in docs),
    }


def evidence_context(docs: list[Document]) -> str:
    """Format evidence chunks with stable chunk IDs for the prompt."""
    blocks: list[str] = []
    for rank, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        blocks.append(
            "\n".join(
                [
                    f"[EVIDENCE {rank}]",
                    f"chunk_id: {meta.get('chunk_id', '')}",
                    f"source: {meta.get('source', '')}",
                    f"title: {meta.get('title', '')}",
                    "text:",
                    doc.page_content,
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "No evidence chunks retrieved."


def llm_from_config(config: dict[str, Any]) -> ChatOllama:
    """Create ChatOllama using only values from configs/config.json."""
    llm_cfg = config.get("LLM", {}) if isinstance(config.get("LLM"), dict) else {}
    model = config["MODELS"]["TEXT_GENERATION"]

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": llm_cfg.get("temperature", config.get("TEMPERATURE", 0)),
    }

    for key in ["num_ctx", "top_p", "repeat_penalty", "num_predict", "seed"]:
        if key in llm_cfg:
            kwargs[key] = llm_cfg[key]

    return ChatOllama(**kwargs)


def build_generation_chains(config: dict[str, Any]) -> dict[str, RunnableSequence]:
    """Build description/pre/post generation chains with improved prompts."""
    llm = llm_from_config(config)

    description_prompt = ChatPromptTemplate.from_template(
        """
You are enriching a CTI attack-step representation.

Use only the cited evidence chunks and the STIX attack-step metadata. Do not use
background knowledge unless it is directly supported by the evidence. Do not
invent tools, infrastructure, privileges, files, registry keys, hosts, or network
indicators that are not present in the evidence.

STIX attack-step metadata:
- action name: {action_name}
- action description: {action_description}
- technique id: {technique_id}
- technique name: {technique_name}
- tactic: {tactic}

BEGIN EVIDENCE CHUNKS
{context}
END EVIDENCE CHUNKS

Task:
Write one concise, formal, technical enriched description of how the action is
performed in this case. The description must be evidence-grounded and specific
to the cited chunks. If the evidence is insufficient, state only what is known
from the STIX metadata and do not add unsupported details.

Output constraints:
- Return only the description text.
- Do not use markdown.
- Do not cite chunk IDs in the text.
- Do not include headings, bullets, or explanations.
""".strip()
    )

    preconditions_prompt = ChatPromptTemplate.from_template(
        """
You are extracting preconditions for a CTI attack step.

A precondition is a technical fact that must plausibly hold before the action can
be performed. Use only the cited evidence chunks and the STIX metadata. Prefer
conditions that are directly stated. Conservative inference is allowed only when
it follows clearly from the evidence. Do not generate generic ATT&CK knowledge.

STIX attack-step metadata:
- action name: {action_name}
- action description: {action_description}
- technique id: {technique_id}
- technique name: {technique_name}
- tactic: {tactic}

BEGIN EVIDENCE CHUNKS
{context}
END EVIDENCE CHUNKS

Task:
List all evidence-grounded preconditions for this action. Do not limit the number
of items. If no precondition is supported or clearly inferable, return an empty
list as [].

Output constraints:
- Return only a numbered list, one precondition per line, or [] if none.
- Each item must be one atomic passive sentence.
- Do not include explanations, headings, markdown, or evidence citations.
- Do not merge multiple conditions into one item.
""".strip()
    )

    postconditions_prompt = ChatPromptTemplate.from_template(
        """
You are extracting postconditions for a CTI attack step.

A postcondition is a technical fact that becomes true, enabled, changed, or left
behind after the action succeeds. Use only the cited evidence chunks and the STIX
metadata. Prefer consequences that are directly stated. Conservative inference is
allowed only when it follows clearly from the evidence. Do not generate generic
ATT&CK knowledge.

STIX attack-step metadata:
- action name: {action_name}
- action description: {action_description}
- technique id: {technique_id}
- technique name: {technique_name}
- tactic: {tactic}

BEGIN EVIDENCE CHUNKS
{context}
END EVIDENCE CHUNKS

Task:
List all evidence-grounded postconditions for this action. Do not limit the
number of items. If no postcondition is supported or clearly inferable, return an
empty list as [].

Output constraints:
- Return only a numbered list, one postcondition per line, or [] if none.
- Each item must be one atomic passive sentence.
- Do not include explanations, headings, markdown, or evidence citations.
- Do not merge multiple conditions into one item.
""".strip()
    )

    return {
        "description": RunnableSequence(
            first=description_prompt,
            middle=[llm],
            last=StrOutputParser(),
        ),
        "preconditions": RunnableSequence(
            first=preconditions_prompt,
            middle=[llm],
            last=StrOutputParser(),
        ),
        "postconditions": RunnableSequence(
            first=postconditions_prompt,
            middle=[llm],
            last=StrOutputParser(),
        ),
    }


def parse_condition_list(raw: str) -> list[str]:
    """Parse numbered/bulleted model output into a list without imposing a limit."""
    cleaned = remove_markdown(raw)
    if not cleaned or cleaned.strip() in {"[]", "[ ]"}:
        return []

    items: list[str] = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        line = line.strip(" \t-–—")
        if not line or line in {"[]", "[ ]"}:
            continue
        # Ignore accidental section labels without content.
        if line.lower() in {"preconditions", "postconditions", "conditions"}:
            continue
        items.append(line)
    return items


def condition_normal_form(text: Any) -> str:
    """Normalize condition text for exact duplicate checks.

    Semantic duplicate removal is handled separately with embeddings. This
    function catches cheap exact duplicates first and makes the embedding stage
    smaller and deterministic.
    """
    text = remove_markdown(str(text or "")).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return squash_ws(text)


def get_duplicate_threshold(config: dict[str, Any]) -> float:
    """Read the semantic duplicate threshold from configs/config.json.

    Supported config locations:
        {"CONDITIONS": {"duplicate_threshold": 0.85}}
        {"DUPLICATE_THRESHOLD": 0.85}

    The threshold follows the paper script convention: a condition is removed
    when its cosine similarity with a previously kept condition is strictly
    greater than the threshold.
    """
    return float(cfg_get(config, "CONDITIONS.duplicate_threshold", config.get("DUPLICATE_THRESHOLD", 0.85)))


def build_duplicate_model(config: dict[str, Any], logger: logging.Logger, exec_log: JsonlLogger) -> Any:
    """Load the embedding model used only for semantic duplicate detection.

    This does not alter the evaluated RAG configuration. It is a post-generation
    cleaning step applied to the conditions generated for one attack step.
    """
    model_name = (
        cfg_get(config, "MODELS.DUPLICATE_EMBEDDING")
        or cfg_get(config, "MODELS.SENTENCE_TRANSFORMER")
        or cfg_get(config, "MODELS.EMBEDDINGS")
    )

    if not model_name:
        logger.warning(
            "Semantic duplicate detection disabled: no MODELS.DUPLICATE_EMBEDDING "
            "or MODELS.SENTENCE_TRANSFORMER configured."
        )
        exec_log.write("duplicate_model_disabled", reason="missing_model_config")
        return None

    if SentenceTransformer is None or st_util is None:
        logger.warning("Semantic duplicate detection disabled: sentence_transformers is unavailable.")
        exec_log.write("duplicate_model_disabled", reason="sentence_transformers_unavailable", model=model_name)
        return None

    try:
        try:
            model = SentenceTransformer(str(model_name), token=os.getenv("HF_API_KEY"))
        except TypeError:
            # Older sentence-transformers versions do not accept the token keyword.
            model = SentenceTransformer(str(model_name))
        logger.info("Loaded duplicate-detection embedding model: %s", model_name)
        exec_log.write("duplicate_model_loaded", model=str(model_name))
        return model
    except Exception as exc:
        logger.exception("Failed loading duplicate-detection embedding model")
        exec_log.write(
            "duplicate_model_error",
            model=str(model_name),
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return None


def deduplicate_conditions_semantically(
    conditions: list[str],
    *,
    threshold: float,
    embedding_model: Any,
    exec_log: JsonlLogger,
    event_context: dict[str, Any],
    condition_type: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Remove exact and semantic duplicate conditions within one attack step.

    The first occurrence is kept. A later condition is removed when it is an
    exact normalized duplicate or when its cosine similarity to a previously
    kept condition is strictly greater than the configured threshold.

    Returns:
        kept_conditions, duplicate_records
    """
    cleaned_conditions = [squash_ws(remove_markdown(c)) for c in conditions]
    cleaned_conditions = [c for c in cleaned_conditions if c]

    duplicate_records: list[dict[str, Any]] = []
    exact_kept: list[str] = []
    exact_norms: dict[str, str] = {}

    for condition in cleaned_conditions:
        norm = condition_normal_form(condition)
        if not norm:
            continue
        if norm in exact_norms:
            duplicate_records.append(
                {
                    **event_context,
                    "condition_type": condition_type,
                    "duplicate_stage": "exact",
                    "removed_statement": condition,
                    "kept_statement": exact_norms[norm],
                    "similarity": 1.0,
                    "threshold": threshold,
                }
            )
            continue
        exact_norms[norm] = condition
        exact_kept.append(condition)

    if len(exact_kept) <= 1 or embedding_model is None:
        exec_log.write(
            "condition_deduplication_finished",
            **event_context,
            condition_type=condition_type,
            input_count=len(conditions),
            output_count=len(exact_kept),
            removed_count=len(duplicate_records),
            threshold=threshold,
            semantic_enabled=embedding_model is not None,
        )
        return exact_kept, duplicate_records

    try:
        embeddings = embedding_model.encode(
            exact_kept,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        kept_conditions: list[str] = []
        kept_indices: list[int] = []

        for idx, condition in enumerate(exact_kept):
            duplicate_of = None
            duplicate_similarity = None

            for kept_idx in kept_indices:
                similarity = float(st_util.cos_sim(embeddings[idx], embeddings[kept_idx]).detach().cpu().item())
                if similarity > threshold:
                    duplicate_of = exact_kept[kept_idx]
                    duplicate_similarity = similarity
                    break

            if duplicate_of is not None:
                duplicate_records.append(
                    {
                        **event_context,
                        "condition_type": condition_type,
                        "duplicate_stage": "semantic",
                        "removed_statement": condition,
                        "kept_statement": duplicate_of,
                        "similarity": duplicate_similarity,
                        "threshold": threshold,
                    }
                )
            else:
                kept_indices.append(idx)
                kept_conditions.append(condition)

        exec_log.write(
            "condition_deduplication_finished",
            **event_context,
            condition_type=condition_type,
            input_count=len(conditions),
            output_count=len(kept_conditions),
            removed_count=len(duplicate_records),
            threshold=threshold,
            semantic_enabled=True,
        )
        return kept_conditions, duplicate_records

    except Exception as exc:
        # Do not fail the pipeline because of a duplicate-checking error. Keep
        # exact-deduplicated conditions and record the failure for transparency.
        exec_log.write(
            "condition_deduplication_error",
            **event_context,
            condition_type=condition_type,
            threshold=threshold,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        logging.getLogger("cti_enrichment").exception("Semantic duplicate detection failed")
        return exact_kept, duplicate_records


def load_stix_steps(config: dict[str, Any], logger: logging.Logger, exec_log: JsonlLogger) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """Parse STIX and group attack patterns by tactic.

    MITRE technique matching is intentionally removed. The technique ID/name are
    extracted from the STIX attack-pattern object itself.
    """
    parser = STIXParser()
    parser.parse(config["STIX_FILE"])

    malware = parser.extract_malware()
    malware_name = malware[0].get("name", "the malware") if malware else "the malware"

    attack_patterns = parser.get_attack_pattern_used()
    if not attack_patterns:
        raise ValueError("No attack-pattern objects found in the STIX file")

    mitre_tactics_file = Path(config.get("MITRE_TACTICS_FILE", "mitre/mitre-tactics.json"))
    with mitre_tactics_file.open("r", encoding="utf-8") as f:
        mitre_tactics = json.load(f)

    grouped = group_attack_patterns(mitre_tactics, attack_patterns)
    logger.info("Loaded %d grouped tactic milestones from STIX", len(grouped))
    exec_log.write("stix_loaded", malware_name=malware_name, attack_steps=sum(len(v) for v in grouped.values()))
    return malware_name, grouped


def technique_from_stix_action(action: dict[str, Any]) -> dict[str, str]:
    """Extract technique metadata directly from a STIX attack-pattern dict."""
    technique_id = str(action.get("technique_id") or action.get("x_mitre_id") or "")
    technique_name = str(action.get("name") or "")
    technique_description = str(action.get("description") or "")

    for ref in action.get("external_references", []) or []:
        if not isinstance(ref, dict):
            continue
        external_id = ref.get("external_id")
        if external_id and not technique_id:
            technique_id = str(external_id)
        if ref.get("source_name") and str(ref.get("source_name")).lower().startswith("mitre"):
            technique_id = technique_id or str(external_id or "")

    return {
        "id": technique_id,
        "name": technique_name,
        "description": technique_description,
    }


def tactic_from_action(action: dict[str, Any], fallback: str) -> str:
    phases = action.get("kill_chain_phases") or []
    if phases and isinstance(phases, list):
        first = phases[0]
        if isinstance(first, dict) and first.get("phase_name"):
            return str(first["phase_name"])
    return str(fallback)


def stable_attack_step_id(case_id: str, tactic: str, index: int, action: dict[str, Any]) -> str:
    # Prefer stable STIX id if available; otherwise derive from case/tactic/name.
    stix_id = str(action.get("id") or "")
    if stix_id:
        return f"{case_id}-{slugify(stix_id)}"
    return f"{case_id}-{slugify(tactic)}-step{index:03d}-{slugify(action.get('name'))}"


def retrieval_query_for_description(action: dict[str, Any], technique: dict[str, str], tactic: str) -> str:
    return squash_ws(
        f"{action.get('name', '')}. {action.get('description', '')}. "
        f"Technique {technique.get('id', '')} {technique.get('name', '')}. Tactic {tactic}. "
        "Evidence describing how this action is performed in the case."
    )


def retrieval_query_for_preconditions(action: dict[str, Any], technique: dict[str, str], enriched_description: str) -> str:
    return squash_ws(
        f"Requirements, prerequisites, environment, access, privileges, tools, resources, or connectivity needed before: "
        f"{action.get('name', '')}. {action.get('description', '')}. {technique.get('id', '')} {technique.get('name', '')}. "
        f"Known enriched description: {enriched_description}"
    )


def retrieval_query_for_postconditions(action: dict[str, Any], technique: dict[str, str], enriched_description: str) -> str:
    return squash_ws(
        f"Consequences, technical traces, files, registry changes, network connections, persistence, access, or enabled capabilities after: "
        f"{action.get('name', '')}. {action.get('description', '')}. {technique.get('id', '')} {technique.get('name', '')}. "
        f"Known enriched description: {enriched_description}"
    )


def generate_text_safely(
    chain: RunnableSequence,
    inputs: dict[str, Any],
    exec_log: JsonlLogger,
    event_context: dict[str, Any],
) -> TimedResult:
    """Run an LLM chain, returning an empty string on failure."""
    try:
        result = timed_call(chain.invoke, inputs)
        text = remove_markdown(result.value)
        exec_log.write("generation_finished", **event_context, latency_ms=result.latency_ms, output=text)
        return TimedResult(value=text, latency_ms=result.latency_ms)
    except Exception as exc:
        exec_log.write(
            "generation_error",
            **event_context,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        logging.getLogger("cti_enrichment").exception("Generation failed: %s", event_context)
        return TimedResult(value="", latency_ms=0.0)


def run_pipeline() -> None:
    dotenv.load_dotenv()
    start = time.perf_counter()
    config = load_config()
    case_id = str(config["CASE_ID"])
    case_output_dir = output_dir_for_case(config)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(case_output_dir)
    exec_log = JsonlLogger(case_output_dir / "execution_log.jsonl")

    try:
        logger.info("Application started for case %s", case_id)
        exec_log.write("run_started", case_id=case_id, config_file=str(CONFIG_FILE))

        # Persist the exact configuration used for reproducibility.
        with (case_output_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        documents = load_documents(config, logger, exec_log)
        write_chunks_jsonl(case_output_dir / "chunks.jsonl", documents)

        rag_config = get_rag_config(config)
        retriever = build_evaluated_hybrid_retriever(documents, rag_config, logger, exec_log)
        chains = build_generation_chains(config)
        duplicate_threshold = get_duplicate_threshold(config)
        duplicate_model = build_duplicate_model(config, logger, exec_log)

        malware_name, grouped_patterns = load_stix_steps(config, logger, exec_log)
        output: list[dict[str, Any]] = []

        retrieval_logs: list[dict[str, Any]] = []
        generation_logs: list[dict[str, Any]] = []
        description_records: list[dict[str, Any]] = []
        condition_records: list[dict[str, Any]] = []
        duplicate_logs: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        attack_step_counter = 0

        for tactic_index, (tactic, actions) in enumerate(grouped_patterns.items(), start=1):
            milestone_id = f"{case_id}-milestone{tactic_index:02d}-{slugify(tactic)}"
            milestone = {
                "id": milestone_id,
                "tactic": tactic,
                "attack_steps": [],
            }
            logger.info("Processing tactic milestone: %s", tactic)
            exec_log.write("milestone_started", milestone_id=milestone_id, tactic=tactic, actions=len(actions))

            for local_index, action in enumerate(actions, start=1):
                attack_step_counter += 1
                action = dict(action)
                action_name = squash_ws(str(action.get("name") or "").replace(malware_name, "the malware"))
                action_description = squash_ws(str(action.get("description") or "").replace(malware_name, "the malware"))
                action["name"] = action_name
                action["description"] = action_description

                attack_step_id = stable_attack_step_id(case_id, tactic, attack_step_counter, action)
                tactic_name = tactic_from_action(action, tactic)
                technique = technique_from_stix_action(action)

                event_base = {
                    "case_id": case_id,
                    "milestone_id": milestone_id,
                    "tactic": tactic_name,
                    "attack_step_id": attack_step_id,
                    "action_name": action_name,
                    "technique_id": technique["id"],
                    "technique_name": technique["name"],
                }
                exec_log.write("attack_step_started", **event_base)

                # Description enrichment.
                desc_query = retrieval_query_for_description(action, technique, tactic_name)
                desc_retrieved = retrieve_evidence(
                    retriever,
                    desc_query,
                    rag_config,
                    exec_log,
                    {**event_base, "field_type": "description"},
                )
                desc_evidence = evidence_payload(desc_retrieved.value)

                desc_inputs = {
                    "action_name": action_name,
                    "action_description": action_description,
                    "technique_id": technique["id"],
                    "technique_name": technique["name"],
                    "tactic": tactic_name,
                    "context": evidence_context(desc_retrieved.value),
                }
                desc_generated = generate_text_safely(
                    chains["description"],
                    desc_inputs,
                    exec_log,
                    {**event_base, "field_type": "description"},
                )
                enriched_description = desc_generated.value

                desc_record = {
                    "field_id": f"{attack_step_id}-desc01",
                    "case_id": case_id,
                    "attack_step_id": attack_step_id,
                    "field_type": "description",
                    "original_description": action_description,
                    "generated_text": enriched_description,
                    **desc_evidence,
                }
                description_records.append(desc_record)
                retrieval_logs.append(
                    {
                        **event_base,
                        "field_type": "description",
                        "query": desc_query,
                        "latency_ms": desc_retrieved.latency_ms,
                        "retrieved_chunks": len(desc_retrieved.value),
                        "evidence_chunk_ids": desc_evidence["evidence_chunk_ids"],
                    }
                )
                generation_logs.append(
                    {
                        **event_base,
                        "field_type": "description",
                        "latency_ms": desc_generated.latency_ms,
                        "input_chunks": len(desc_retrieved.value),
                        "context_tokens": desc_evidence["evidence_token_count"],
                        "output_tokens": token_count(enriched_description),
                        "success": bool(enriched_description),
                    }
                )
                if not enriched_description:
                    errors.append({**event_base, "stage": "description_generation", "error": "empty_generation"})

                # Preconditions.
                pre_query = retrieval_query_for_preconditions(action, technique, enriched_description)
                pre_retrieved = retrieve_evidence(
                    retriever,
                    pre_query,
                    rag_config,
                    exec_log,
                    {**event_base, "field_type": "preconditions"},
                )
                pre_evidence = evidence_payload(pre_retrieved.value)
                pre_inputs = {
                    "action_name": action_name,
                    "action_description": action_description,
                    "technique_id": technique["id"],
                    "technique_name": technique["name"],
                    "tactic": tactic_name,
                    "context": evidence_context(pre_retrieved.value),
                }
                pre_generated = generate_text_safely(
                    chains["preconditions"],
                    pre_inputs,
                    exec_log,
                    {**event_base, "field_type": "preconditions"},
                )
                preconditions = parse_condition_list(pre_generated.value)
                preconditions_raw_count = len(preconditions)
                if pre_generated.value and not preconditions:
                    errors.append({**event_base, "stage": "precondition_parsing", "error": "no_items_parsed"})

                preconditions, precondition_duplicates = deduplicate_conditions_semantically(
                    preconditions,
                    threshold=duplicate_threshold,
                    embedding_model=duplicate_model,
                    exec_log=exec_log,
                    event_context=event_base,
                    condition_type="precondition",
                )
                duplicate_logs.extend(precondition_duplicates)

                # Postconditions.
                post_query = retrieval_query_for_postconditions(action, technique, enriched_description)
                post_retrieved = retrieve_evidence(
                    retriever,
                    post_query,
                    rag_config,
                    exec_log,
                    {**event_base, "field_type": "postconditions"},
                )
                post_evidence = evidence_payload(post_retrieved.value)
                post_inputs = {
                    "action_name": action_name,
                    "action_description": action_description,
                    "technique_id": technique["id"],
                    "technique_name": technique["name"],
                    "tactic": tactic_name,
                    "context": evidence_context(post_retrieved.value),
                }
                post_generated = generate_text_safely(
                    chains["postconditions"],
                    post_inputs,
                    exec_log,
                    {**event_base, "field_type": "postconditions"},
                )
                postconditions = parse_condition_list(post_generated.value)
                postconditions_raw_count = len(postconditions)
                if post_generated.value and not postconditions:
                    errors.append({**event_base, "stage": "postcondition_parsing", "error": "no_items_parsed"})

                postconditions, postcondition_duplicates = deduplicate_conditions_semantically(
                    postconditions,
                    threshold=duplicate_threshold,
                    embedding_model=duplicate_model,
                    exec_log=exec_log,
                    event_context=event_base,
                    condition_type="postcondition",
                )
                duplicate_logs.extend(postcondition_duplicates)

                generation_logs.extend(
                    [
                        {
                            **event_base,
                            "field_type": "preconditions",
                            "latency_ms": pre_generated.latency_ms,
                            "input_chunks": len(pre_retrieved.value),
                            "context_tokens": pre_evidence["evidence_token_count"],
                            "output_tokens": token_count(pre_generated.value),
                            "success": bool(pre_generated.value),
                            "parsed_items_raw": preconditions_raw_count,
                            "semantic_duplicates_removed": len(precondition_duplicates),
                            "parsed_items": len(preconditions),
                        },
                        {
                            **event_base,
                            "field_type": "postconditions",
                            "latency_ms": post_generated.latency_ms,
                            "input_chunks": len(post_retrieved.value),
                            "context_tokens": post_evidence["evidence_token_count"],
                            "output_tokens": token_count(post_generated.value),
                            "success": bool(post_generated.value),
                            "parsed_items_raw": postconditions_raw_count,
                            "semantic_duplicates_removed": len(postcondition_duplicates),
                            "parsed_items": len(postconditions),
                        },
                    ]
                )
                retrieval_logs.extend(
                    [
                        {
                            **event_base,
                            "field_type": "preconditions",
                            "query": pre_query,
                            "latency_ms": pre_retrieved.latency_ms,
                            "retrieved_chunks": len(pre_retrieved.value),
                            "evidence_chunk_ids": pre_evidence["evidence_chunk_ids"],
                        },
                        {
                            **event_base,
                            "field_type": "postconditions",
                            "query": post_query,
                            "latency_ms": post_retrieved.latency_ms,
                            "retrieved_chunks": len(post_retrieved.value),
                            "evidence_chunk_ids": post_evidence["evidence_chunk_ids"],
                        },
                    ]
                )

                pre_records = []
                for i, statement in enumerate(preconditions, start=1):
                    record = {
                        "condition_id": f"{attack_step_id}-pre{i:02d}",
                        "case_id": case_id,
                        "attack_step_id": attack_step_id,
                        "condition_type": "precondition",
                        "statement": statement,
                        # Evidence is attached deterministically from the RAG chunks used
                        # to generate this condition. The model does not choose evidence.
                        **pre_evidence,
                        "evidence": pre_evidence,
                    }
                    condition_records.append(record)
                    pre_records.append(record)

                post_records = []
                for i, statement in enumerate(postconditions, start=1):
                    record = {
                        "condition_id": f"{attack_step_id}-post{i:02d}",
                        "case_id": case_id,
                        "attack_step_id": attack_step_id,
                        "condition_type": "postcondition",
                        "statement": statement,
                        # Evidence is attached deterministically from the RAG chunks used
                        # to generate this condition. The model does not choose evidence.
                        **post_evidence,
                        "evidence": post_evidence,
                    }
                    condition_records.append(record)
                    post_records.append(record)

                milestone["attack_steps"].append(
                    {
                        "id": attack_step_id,
                        "name": action_name,
                        "original_description": action_description,
                        "description": enriched_description,
                        "mitre_technique": technique,
                        "tactic": tactic_name,
                        "description_evidence_chunk_ids": desc_evidence["evidence_chunk_ids"],
                        "pre-conditions": [r["statement"] for r in pre_records],
                        "post-conditions": [r["statement"] for r in post_records],
                        # Condition objects preserve per-condition evidence for
                        # annotation and traceability. The flat string lists above
                        # are retained for compatibility with the original script.
                        "precondition_objects": pre_records,
                        "postcondition_objects": post_records,
                        "precondition_evidence_chunk_ids": pre_evidence["evidence_chunk_ids"],
                        "postcondition_evidence_chunk_ids": post_evidence["evidence_chunk_ids"],
                    }
                )

                exec_log.write(
                    "attack_step_finished",
                    **event_base,
                    description_generated=bool(enriched_description),
                    preconditions=len(preconditions),
                    postconditions=len(postconditions),
                    precondition_duplicates_removed=len(precondition_duplicates),
                    postcondition_duplicates_removed=len(postcondition_duplicates),
                )

            output.append(milestone)
            exec_log.write("milestone_finished", milestone_id=milestone_id, attack_steps=len(milestone["attack_steps"]))

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics = compute_metrics(
            config=config,
            rag_config=rag_config,
            documents=documents,
            output=output,
            retrieval_logs=retrieval_logs,
            generation_logs=generation_logs,
            description_records=description_records,
            condition_records=condition_records,
            duplicate_logs=duplicate_logs,
            errors=errors,
            elapsed_ms=elapsed_ms,
        )

        output_file = str(config.get("OUTPUT_FILE", "enriched_attack_graph.json"))
        write_json(case_output_dir / output_file, output)
        write_json(case_output_dir / "metrics.json", metrics)
        write_json(case_output_dir / "errors.json", errors)
        write_jsonl(case_output_dir / "retrieval_logs.jsonl", retrieval_logs)
        write_jsonl(case_output_dir / "generation_logs.jsonl", generation_logs)
        write_jsonl(case_output_dir / "generated_descriptions.jsonl", description_records)
        write_jsonl(case_output_dir / "generated_conditions.jsonl", condition_records)
        write_jsonl(case_output_dir / "final_conditions.jsonl", condition_records)
        write_jsonl(case_output_dir / "condition_duplicate_logs.jsonl", duplicate_logs)

        exec_log.write("run_finished", elapsed_ms=elapsed_ms, output_file=output_file, errors=len(errors))
        logger.info("Application finished in %.2f minutes", elapsed_ms / 60000.0)

    except Exception as exc:
        exec_log.write("run_crashed", error=str(exc), traceback=traceback.format_exc())
        logger.exception("Application crashed")
        raise
    finally:
        exec_log.close()


def compute_metrics(
    *,
    config: dict[str, Any],
    rag_config: dict[str, Any],
    documents: list[Document],
    output: list[dict[str, Any]],
    retrieval_logs: list[dict[str, Any]],
    generation_logs: list[dict[str, Any]],
    description_records: list[dict[str, Any]],
    condition_records: list[dict[str, Any]],
    duplicate_logs: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    elapsed_ms: float,
) -> dict[str, Any]:
    attack_steps = [step for milestone in output for step in milestone.get("attack_steps", [])]
    preconditions = [c for c in condition_records if c.get("condition_type") == "precondition"]
    postconditions = [c for c in condition_records if c.get("condition_type") == "postcondition"]

    non_empty_descriptions = [d for d in description_records if squash_ws(d.get("generated_text"))]
    steps_with_pre = {c["attack_step_id"] for c in preconditions if squash_ws(c.get("statement"))}
    steps_with_post = {c["attack_step_id"] for c in postconditions if squash_ws(c.get("statement"))}

    def safe_rate(num: int, den: int) -> float | None:
        return None if den == 0 else num / den

    def avg(values: Iterable[float]) -> float | None:
        values = list(values)
        return None if not values else sum(values) / len(values)

    def avg_len(records: list[dict[str, Any]], key: str) -> float | None:
        return avg(token_count(r.get(key, "")) for r in records)

    def avg_evidence(records: list[dict[str, Any]]) -> float | None:
        return avg(len(r.get("evidence_chunk_ids", []) or []) for r in records)

    retrieval_latencies = [float(r.get("latency_ms", 0.0)) for r in retrieval_logs]
    generation_latencies = [float(r.get("latency_ms", 0.0)) for r in generation_logs]
    context_tokens = [float(r.get("context_tokens", 0.0)) for r in generation_logs]
    duplicate_pre = [d for d in duplicate_logs if d.get("condition_type") == "precondition"]
    duplicate_post = [d for d in duplicate_logs if d.get("condition_type") == "postcondition"]

    return {
        "case_id": config["CASE_ID"],
        "run_timestamp": now_iso(),
        "pipeline_config": {
            "models": config.get("MODELS", {}),
            "rag": rag_config,
            "stix_file": config.get("STIX_FILE"),
            "documents_dir": config.get("DOCUMENTS_DIR"),
            "document_subdir": config.get("DOCUMENTS_SUBDIR"),
        },
        "counts": {
            "documents_or_chunks": len(documents),
            "milestones": len(output),
            "attack_steps": len(attack_steps),
            "generated_descriptions": len(non_empty_descriptions),
            "generated_preconditions": len(preconditions),
            "generated_postconditions": len(postconditions),
            "generated_conditions_total": len(condition_records),
            "semantic_duplicate_preconditions_removed": len(duplicate_pre),
            "semantic_duplicate_postconditions_removed": len(duplicate_post),
            "semantic_duplicate_conditions_removed": len(duplicate_logs),
            "errors": len(errors),
        },
        "automatic_metrics": {
            "field_coverage": {
                "descriptions": safe_rate(len(non_empty_descriptions), len(attack_steps)),
                "preconditions": safe_rate(len(steps_with_pre), len(attack_steps)),
                "postconditions": safe_rate(len(steps_with_post), len(attack_steps)),
                "preconditions_and_postconditions": safe_rate(len(steps_with_pre & steps_with_post), len(attack_steps)),
            },
            "evidence_link_coverage": {
                "descriptions": safe_rate(sum(bool(r.get("evidence_chunk_ids")) for r in description_records), len(description_records)),
                "preconditions": safe_rate(sum(bool(r.get("evidence_chunk_ids")) for r in preconditions), len(preconditions)),
                "postconditions": safe_rate(sum(bool(r.get("evidence_chunk_ids")) for r in postconditions), len(postconditions)),
            },
            "average_evidence_count": {
                "descriptions": avg_evidence(description_records),
                "preconditions": avg_evidence(preconditions),
                "postconditions": avg_evidence(postconditions),
            },
            "average_field_length_tokens": {
                "descriptions": avg_len(description_records, "generated_text"),
                "preconditions": avg_len(preconditions, "statement"),
                "postconditions": avg_len(postconditions, "statement"),
            },
            "duplicate_condition_rate": {
                "threshold": get_duplicate_threshold(config),
                "preconditions": safe_rate(len(duplicate_pre), len(preconditions) + len(duplicate_pre)),
                "postconditions": safe_rate(len(duplicate_post), len(postconditions) + len(duplicate_post)),
                "total": safe_rate(len(duplicate_logs), len(condition_records) + len(duplicate_logs)),
                "removed_preconditions": len(duplicate_pre),
                "removed_postconditions": len(duplicate_post),
                "removed_total": len(duplicate_logs),
            },
            "latency_ms": {
                "average_retrieval": avg(retrieval_latencies),
                "average_generation": avg(generation_latencies),
                "total_retrieval": sum(retrieval_latencies),
                "total_generation": sum(generation_latencies),
                "total_pipeline": elapsed_ms,
            },
            "retrieval_transferability": {
                "average_retrieved_context_tokens": avg(context_tokens),
                "average_retrieved_chunks_per_call": avg(r.get("retrieved_chunks", 0) for r in retrieval_logs),
            },
            "generation_failures": {
                "error_count": len(errors),
                "error_rate_per_attack_step": safe_rate(len(errors), len(attack_steps)),
            },
        },
        "notes": [
            "No verifier is used in this pipeline version.",
            "Evidence is deterministically assigned from the retrieved RAG chunks.",
            "The model does not assign support/confidence/status labels.",
            "Semantic duplicate conditions are removed when cosine similarity is greater than the configured threshold.",
        ],
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    run_pipeline()
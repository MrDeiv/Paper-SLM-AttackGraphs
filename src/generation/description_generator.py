"""
Evidence-grounded description generation for attack steps.

The model generates only the description text. The pipeline deterministically
attaches the evidence chunk IDs returned by the frozen RAG retriever.
"""

from __future__ import annotations

from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

from src.generation.condition_generator import format_chunks_for_prompt
from src.generation.json_parser import JSONParseError, parse_json_object
from src.generation.prompts import DESCRIPTION_PROMPT_V2
from src.schemas import AttackStepRecord, GeneratedDescription


class DescriptionGenerator:
    """Generate an enriched description for one attack step."""

    def __init__(
        self,
        *,
        model_id: str,
        temperature: float = 0.0,
        prompt_version: str = "description_v3_text_only",
        num_ctx: int | None = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.prompt_version = prompt_version
        self.num_ctx = num_ctx

        # num_ctx is passed to Ollama but does not alter retrieval. It only
        # controls the local model runtime context size.
        llm_kwargs: dict[str, Any] = {"model": model_id, "temperature": temperature}
        if num_ctx is not None:
            llm_kwargs["num_ctx"] = int(num_ctx)
        self.llm = ChatOllama(**llm_kwargs)
        self.chain = RunnableSequence(
            first=ChatPromptTemplate.from_template(DESCRIPTION_PROMPT_V2),
            middle=[self.llm],
            last=StrOutputParser(),
        )

    def generate(
        self,
        *,
        run_id: str,
        case_id: str,
        attack_step: AttackStepRecord,
        retrieved_chunks: list[dict[str, Any]],
    ) -> GeneratedDescription:
        """Generate a description and attach all retrieved evidence chunks.

        Evidence is deliberately deterministic: every chunk passed to the model
        is recorded as evidence for the generated field. The model is not asked
        to select or score evidence.
        """
        raw_output = self.chain.invoke(
            {
                "attack_step": _format_attack_step(attack_step),
                "evidence_chunks": format_chunks_for_prompt(retrieved_chunks),
            }
        )

        description = _parse_description(raw_output)
        evidence_ids = evidence_chunk_ids(retrieved_chunks)
        source_documents = source_documents_for_chunks(evidence_ids, retrieved_chunks)

        return GeneratedDescription(
            run_id=run_id,
            case_id=case_id,
            attack_step_id=attack_step.attack_step_id,
            description=description,
            evidence_chunk_ids=evidence_ids,
            source_documents=source_documents,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            generation_params={"temperature": self.temperature, "num_ctx": self.num_ctx},
        )


def _parse_description(raw_output: str) -> str:
    try:
        payload = parse_json_object(raw_output)
        return " ".join(str(payload.get("description", "")).split())
    except JSONParseError:
        # Fallback keeps the pipeline robust if the model emits text instead of
        # JSON. The caller can still inspect the raw text as the description.
        return " ".join(str(raw_output or "").split())


def _format_attack_step(step: AttackStepRecord) -> str:
    return "\n".join(
        [
            f"attack_step_id: {step.attack_step_id}",
            f"tactic: {step.attack_tactic}",
            f"label: {step.original_label}",
            f"description: {step.input_text}",
        ]
    )


def evidence_chunk_ids(chunks: list[dict[str, Any]]) -> list[str]:
    """Return chunk IDs in retrieved-rank order, without duplicates."""
    ids: list[str] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if chunk_id and chunk_id not in ids:
            ids.append(chunk_id)
    return ids


def source_documents_for_chunks(ids: list[str], chunks: list[dict[str, Any]]) -> list[str]:
    """Return source document IDs for the selected chunk IDs."""
    chunk_map = {str(chunk.get("chunk_id", "")): chunk for chunk in chunks}
    docs: list[str] = []
    for chunk_id in ids:
        doc_id = str(chunk_map.get(chunk_id, {}).get("document_id", "")).strip()
        if doc_id and doc_id not in docs:
            docs.append(doc_id)
    return docs

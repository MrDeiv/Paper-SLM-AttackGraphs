"""
Structured pre/post-condition generation for attack steps.

The model generates condition statements only. The pipeline constructs the final
JSON records, assigns deterministic IDs, and attaches evidence chunk IDs from the
frozen RAG retriever.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

from src.generation.json_parser import JSONParseError, parse_json_object
from src.generation.prompts import CONDITION_PROMPT_V2
from src.id_utils import make_condition_id
from src.schemas import AttackStepRecord, GeneratedCondition, normalize_condition_dict, normalize_condition_statement


class ConditionGenerator:
    """Generate preconditions and postconditions for one attack step."""

    def __init__(
        self,
        *,
        model_id: str,
        temperature: float = 0.0,
        prompt_version: str = "conditions_v3_text_only",
        num_ctx: int | None = None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.prompt_version = prompt_version
        self.num_ctx = num_ctx

        # num_ctx controls Ollama runtime only. It does not change retrieval.
        llm_kwargs: dict[str, Any] = {"model": model_id, "temperature": temperature}
        if num_ctx is not None:
            llm_kwargs["num_ctx"] = int(num_ctx)
        self.llm = ChatOllama(**llm_kwargs)
        self.chain = RunnableSequence(
            first=ChatPromptTemplate.from_template(CONDITION_PROMPT_V2),
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
        max_conditions: int | None = None,
    ) -> list[GeneratedCondition]:
        """Generate and normalize conditions.

        max_conditions is accepted only for backward compatibility. It is ignored
        by default because the production pipeline must not impose a hard cap on
        the number of generated conditions.
        """
        raw_output = self.chain.invoke(
            {
                "attack_step": _format_attack_step(attack_step),
                "evidence_chunks": format_chunks_for_prompt(retrieved_chunks),
            }
        )

        raw_conditions = _extract_conditions(raw_output)
        evidence_ids = _evidence_chunk_ids(retrieved_chunks)
        source_documents = _source_documents_for_chunks(evidence_ids, retrieved_chunks)

        conditions: list[GeneratedCondition] = []
        counters = {"pre": 0, "post": 0}

        for raw in raw_conditions:
            try:
                normalized = normalize_condition_dict(raw)
            except ValueError:
                continue

            condition_type = normalized["condition_type"]
            counters[condition_type] += 1
            condition_id = make_condition_id(
                case_id,
                attack_step.attack_step_id,
                condition_type,
                counters[condition_type],
            )

            conditions.append(
                GeneratedCondition(
                    condition_id=condition_id,
                    run_id=run_id,
                    case_id=case_id,
                    attack_step_id=attack_step.attack_step_id,
                    condition_type=condition_type,
                    statement=normalized["statement"],
                    evidence_chunk_ids=evidence_ids,
                    source_documents=source_documents,
                    model_id=self.model_id,
                    prompt_version=self.prompt_version,
                    generation_params={"temperature": self.temperature, "num_ctx": self.num_ctx},
                )
            )

        return conditions


def format_chunks_for_prompt(
    chunks: list[dict[str, Any]],
    max_chars_per_chunk: int | None = None,
) -> str:
    """Format retrieved chunks as evidence blocks for prompting.

    By default, chunk text is not truncated. The optional max_chars_per_chunk is
    retained only for backward compatibility with older scripts.
    """
    if not chunks:
        return "No evidence chunks provided."

    blocks: list[str] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", "missing_chunk_id"))
        source = str(chunk.get("source_path", chunk.get("source", "unknown_source")))
        rank = chunk.get("rank", "")
        text = str(chunk.get("text", ""))
        if max_chars_per_chunk is not None:
            text = text[: max(0, int(max_chars_per_chunk))]
        rank_line = f"[rank: {rank}]\n" if rank != "" else ""
        blocks.append(f"{rank_line}[chunk_id: {chunk_id}]\n[source: {source}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def _format_attack_step(step: AttackStepRecord) -> str:
    return "\n".join(
        [
            f"attack_step_id: {step.attack_step_id}",
            f"tactic: {step.attack_tactic}",
            f"label: {step.original_label}",
            f"description: {step.input_text}",
        ]
    )


def _extract_conditions(raw_output: str) -> list[dict[str, Any]]:
    """Extract raw conditions from either the new or legacy JSON schema."""
    try:
        payload = parse_json_object(raw_output)
    except JSONParseError:
        return []

    raw_conditions: list[dict[str, Any]] = []

    # Preferred schema: {"preconditions": [...], "postconditions": [...]}
    for key, condition_type in [("preconditions", "pre"), ("postconditions", "post")]:
        values = payload.get(key, [])
        if isinstance(values, str):
            values = [values]
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict):
                    statement = value.get("statement") or value.get("text") or value.get("condition")
                else:
                    statement = value
                statement = normalize_condition_statement(statement)
                if statement:
                    raw_conditions.append({"condition_type": condition_type, "statement": statement})

    # Backward-compatible schema: {"conditions": [{...}, ...]}
    legacy = payload.get("conditions", [])
    if isinstance(legacy, list):
        for value in legacy:
            if isinstance(value, dict):
                raw_conditions.append(value)

    return _deduplicate_conditions(raw_conditions)


def _deduplicate_conditions(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for condition in conditions:
        condition_type = str(condition.get("condition_type", "")).strip().lower()
        statement = normalize_condition_statement(condition.get("statement"))
        key = (condition_type, statement.lower())
        if condition_type and statement and key not in seen:
            seen.add(key)
            output.append({"condition_type": condition_type, "statement": statement})
    return output


def _evidence_chunk_ids(chunks: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if chunk_id and chunk_id not in ids:
            ids.append(chunk_id)
    return ids


def _source_documents_for_chunks(ids: list[str], chunks: list[dict[str, Any]]) -> list[str]:
    chunk_map = {str(chunk.get("chunk_id", "")): chunk for chunk in chunks}
    docs: list[str] = []
    for chunk_id in ids:
        doc_id = str(chunk_map.get(chunk_id, {}).get("document_id", "")).strip()
        if doc_id and doc_id not in docs:
            docs.append(doc_id)
    return docs


def condition_to_prompt_dict(condition: GeneratedCondition) -> dict[str, Any]:
    return asdict(condition)

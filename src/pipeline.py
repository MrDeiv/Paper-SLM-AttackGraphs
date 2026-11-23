"""
Production CTI enrichment pipeline orchestration.

The pipeline implements the final design agreed for the work:

1. load the pipeline configuration;
2. load the intermediate representation, STIX in the current case-study setup;
3. load, chunk, and store CTI documents in the frozen RAG component;
4. for each extracted attack step, retrieve evidence, generate an enriched
   description, and generate pre/postconditions;
5. attach evidence deterministically from the retrieved RAG chunks;
6. emit graph-ready artifacts without verifier/support/confidence labels.

The retrieval implementation is deliberately delegated to src.retrieval so the
same evaluated RAG component is used unchanged.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - yaml is expected in the project env
    yaml = None  # type: ignore[assignment]

from langchain_core.documents import Document

from src.DocumentFactory import DocumentFactory
from src.attack_steps import attack_step_query, load_attack_steps_from_stix
from src.generation.condition_generator import ConditionGenerator
from src.generation.description_generator import DescriptionGenerator
from src.graph.temporal_milestone_graph import build_temporal_milestone_graph
from src.io_utils import ensure_dir, write_json, write_jsonl
from src.retrieval import build_retriever, retrieve_with_provenance
from src.schemas import (
    AttackStepRecord,
    GeneratedCondition,
    GeneratedDescription,
    RetrievalRecord,
    RuntimeRecord,
    to_dict,
)


@dataclass
class PipelineArtifacts:
    """In-memory artifacts produced by one case-study run."""

    run_id: str
    case_id: str
    config: dict[str, Any]
    attack_steps: list[AttackStepRecord] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    retrieval_results: list[RetrievalRecord] = field(default_factory=list)
    generated_descriptions: list[GeneratedDescription] = field(default_factory=list)
    generated_conditions: list[GeneratedCondition] = field(default_factory=list)
    runtime_metrics: list[RuntimeRecord] = field(default_factory=list)
    enriched_graph: dict[str, Any] | None = None


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON pipeline configuration."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML configs")
        data = yaml.safe_load(text)
    else:
        import json

        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Pipeline config must contain an object: {path}")
    return data


def run_case_from_config(config_path: str | Path, *, case_id: str | None = None) -> PipelineArtifacts:
    """Load a config file and run one configured case."""
    config = load_pipeline_config(config_path)
    return run_case(config, case_id=case_id)


def run_case(config: dict[str, Any], *, case_id: str | None = None) -> PipelineArtifacts:
    """Run the production enrichment pipeline for one case.

    The function is intentionally config-driven. It does not hardcode RAG
    parameters; it forwards the config to src.retrieval.build_retriever.
    """
    run_id = str(config.get("run", {}).get("run_id", "run"))
    case_cfg = _resolve_case_config(config, case_id)
    case_id = str(case_cfg["case_id"])

    attack_steps = load_attack_steps_from_stix(
        str(_required_path(case_cfg, "stix_file")),
        case_id=case_id,
        mitre_tactics_file=str(_path_from_config(case_cfg.get("mitre_tactics_file", "mitre/mitre-tactics.json"))),
    )

    documents = load_and_chunk_documents(config, case_cfg)
    if not documents:
        raise ValueError(f"No evidence chunks were produced for case_id={case_id!r}")

    chunks = [_chunk_document_to_record(doc) for doc in documents]
    retriever = build_retriever(documents, config)

    text_model = _text_model_from_config(config)
    generation_cfg = dict(config.get("generation", {}))
    llm_runtime = dict(config.get("llm_runtime", {}))
    temperature = float(generation_cfg.get("temperature", 0.0))
    num_ctx = llm_runtime.get("num_ctx", generation_cfg.get("num_ctx"))
    num_ctx = int(num_ctx) if num_ctx is not None else None

    description_generator = DescriptionGenerator(
        model_id=text_model,
        temperature=temperature,
        num_ctx=num_ctx,
    )
    condition_generator = ConditionGenerator(
        model_id=text_model,
        temperature=temperature,
        num_ctx=num_ctx,
    )

    artifacts = PipelineArtifacts(
        run_id=run_id,
        case_id=case_id,
        config=config,
        attack_steps=attack_steps,
        chunks=chunks,
    )

    for step in attack_steps:
        query = attack_step_query(step)

        t0 = time.perf_counter()
        retrieval_record = retrieve_with_provenance(
            retriever,
            query=query,
            run_id=run_id,
            case_id=case_id,
            attack_step_id=step.attack_step_id,
        )
        artifacts.retrieval_results.append(retrieval_record)
        artifacts.runtime_metrics.append(
            RuntimeRecord(
                run_id=run_id,
                case_id=case_id,
                attack_step_id=step.attack_step_id,
                stage="retrieval",
                elapsed_ms=_elapsed_ms(t0),
                metadata={"retriever": retrieval_record.retriever, "top_k": retrieval_record.top_k},
            )
        )

        retrieved_chunks = [to_dict(result) for result in retrieval_record.results]

        t0 = time.perf_counter()
        description = description_generator.generate(
            run_id=run_id,
            case_id=case_id,
            attack_step=step,
            retrieved_chunks=retrieved_chunks,
        )
        artifacts.generated_descriptions.append(description)
        artifacts.runtime_metrics.append(
            RuntimeRecord(
                run_id=run_id,
                case_id=case_id,
                attack_step_id=step.attack_step_id,
                stage="description_generation",
                elapsed_ms=_elapsed_ms(t0),
                model_id=text_model,
            )
        )

        t0 = time.perf_counter()
        conditions = condition_generator.generate(
            run_id=run_id,
            case_id=case_id,
            attack_step=step,
            retrieved_chunks=retrieved_chunks,
            max_conditions=None,
        )
        artifacts.generated_conditions.extend(conditions)
        artifacts.runtime_metrics.append(
            RuntimeRecord(
                run_id=run_id,
                case_id=case_id,
                attack_step_id=step.attack_step_id,
                stage="condition_generation",
                elapsed_ms=_elapsed_ms(t0),
                model_id=text_model,
                metadata={"num_conditions": len(conditions)},
            )
        )

    artifacts.enriched_graph = build_temporal_milestone_graph(
        run_id=run_id,
        case_id=case_id,
        attack_steps=[to_dict(step) for step in artifacts.attack_steps],
        descriptions=[to_dict(desc) for desc in artifacts.generated_descriptions],
        final_conditions=[to_dict(cond) for cond in artifacts.generated_conditions],
        retrieval_results=[to_dict(record) for record in artifacts.retrieval_results],
        chunks=artifacts.chunks,
        config=config,
    )
    return artifacts


def load_and_chunk_documents(config: dict[str, Any], case_cfg: dict[str, Any]) -> list[Document]:
    """Load evidence documents and chunk them with the configured chunker."""
    case_id = str(case_cfg["case_id"])
    chunking_cfg = dict(config.get("chunking", {}))
    chunking_strategy = str(chunking_cfg.get("strategy", "semantic"))
    chunking_config = dict(chunking_cfg.get("config", {}))
    # Also allow flat chunking keys such as max_chunk_tokens/chunk_size.
    for key, value in chunking_cfg.items():
        if key not in {"strategy", "config"}:
            chunking_config.setdefault(key, value)

    documents: list[Document] = []

    for file_path in _case_document_files(case_cfg):
        documents.extend(
            DocumentFactory.from_file(
                str(file_path),
                case_id=case_id,
                chunking_strategy=chunking_strategy,
                chunking_config=chunking_config,
            )
        )

    for directory in _case_document_dirs(case_cfg):
        documents.extend(
            DocumentFactory.from_directory(
                str(directory),
                case_id=case_id,
                chunking_strategy=chunking_strategy,
                chunking_config=chunking_config,
                recursive=True,
            )
        )

    return documents


def write_case_artifacts(artifacts: PipelineArtifacts, output_dir: str | Path) -> dict[str, str]:
    """Persist graph-ready pipeline artifacts to a case output directory."""
    output_dir = ensure_dir(output_dir)
    paths = {
        "config": output_dir / "config.json",
        "attack_steps": output_dir / "attack_steps.jsonl",
        "chunks": output_dir / "chunks.jsonl",
        "retrieval_results": output_dir / "retrieval_results.jsonl",
        "generated_descriptions": output_dir / "generated_descriptions.jsonl",
        "generated_conditions": output_dir / "generated_conditions.jsonl",
        "final_conditions": output_dir / "final_conditions.jsonl",
        "runtime_metrics": output_dir / "runtime_metrics.jsonl",
        "enriched_graph": output_dir / "enriched_graph.json",
    }

    write_json(artifacts.config, paths["config"])
    write_jsonl([to_dict(step) for step in artifacts.attack_steps], paths["attack_steps"])
    write_jsonl(artifacts.chunks, paths["chunks"])
    write_jsonl([to_dict(record) for record in artifacts.retrieval_results], paths["retrieval_results"])
    write_jsonl([to_dict(desc) for desc in artifacts.generated_descriptions], paths["generated_descriptions"])
    write_jsonl([to_dict(cond) for cond in artifacts.generated_conditions], paths["generated_conditions"])
    # final_conditions is intentionally identical to generated_conditions because
    # the production pipeline has no verifier stage.
    write_jsonl([to_dict(cond) for cond in artifacts.generated_conditions], paths["final_conditions"])
    write_jsonl([to_dict(metric) for metric in artifacts.runtime_metrics], paths["runtime_metrics"])
    write_json(artifacts.enriched_graph or {}, paths["enriched_graph"])

    return {name: str(path) for name, path in paths.items()}


def _resolve_case_config(config: dict[str, Any], case_id: str | None) -> dict[str, Any]:
    cases = config.get("cases") or config.get("case_studies") or []
    if isinstance(cases, dict):
        cases = [{"case_id": key, **value} if isinstance(value, dict) else {"case_id": key} for key, value in cases.items()]
    if not isinstance(cases, list) or not cases:
        raise ValueError("Config must contain a non-empty 'cases' list or mapping")

    if case_id is None:
        if len(cases) != 1:
            raise ValueError("case_id must be provided when the config contains multiple cases")
        case = cases[0]
        if not isinstance(case, dict):
            raise ValueError("Case entries must be dictionaries")
        return case

    for case in cases:
        if isinstance(case, dict) and str(case.get("case_id")) == str(case_id):
            return case
    raise ValueError(f"case_id={case_id!r} not found in config")


def _case_document_files(case_cfg: dict[str, Any]) -> list[Path]:
    values: list[Any] = []
    for key in ["documents", "document_files", "files"]:
        value = case_cfg.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value:
            values.append(value)
    return [_path_from_config(value) for value in values]


def _case_document_dirs(case_cfg: dict[str, Any]) -> list[Path]:
    values: list[Any] = []
    for key in ["documents_dir", "document_dir", "corpus_dir"]:
        value = case_cfg.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value:
            values.append(value)
    return [_path_from_config(value) for value in values]


def _required_path(mapping: dict[str, Any], key: str) -> Path:
    value = mapping.get(key)
    if not value:
        raise ValueError(f"Case config must define {key!r}")
    return _path_from_config(value)


def _path_from_config(value: Any) -> Path:
    return Path(str(value)).expanduser()


def _chunk_document_to_record(doc: Document) -> dict[str, Any]:
    metadata = dict(doc.metadata or {})
    return {
        "case_id": str(metadata.get("case_id", "")),
        "document_id": str(metadata.get("document_id", "")),
        "chunk_id": str(metadata.get("chunk_id", "")),
        "chunk_index": metadata.get("chunk_index"),
        "text": doc.page_content,
        "token_count": metadata.get("token_count"),
        "source": metadata.get("source", metadata.get("source_path", "")),
        "title": metadata.get("title", ""),
        "source_path": metadata.get("source_path", metadata.get("source", "")),
        "chunking_strategy": metadata.get("chunking_strategy", ""),
        "metadata": metadata,
    }


def _text_model_from_config(config: dict[str, Any]) -> str:
    models = dict(config.get("models", {}))
    for key in ["text_generation", "slm_model", "model", "llm"]:
        value = models.get(key)
        if value:
            return str(value)
    generation = dict(config.get("generation", {}))
    if generation.get("model"):
        return str(generation["model"])
    raise ValueError("Config must define models.text_generation or models.slm_model")


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0

"""
Temporal milestone DAG construction for enriched attack graphs.

The graph model is deliberately conservative:

- exactly one START node and one END node per case;
- milestone nodes represent tactic-aligned temporal phases;
- every milestone has at least one attack step;
- every attack step belongs to exactly one milestone;
- attack steps inside a milestone are not ordered by default;
- descriptions, conditions, and evidence are attack-step properties, not nodes;
- visual edges are unlabeled by default.

For each milestone M_i with attack steps {a_i1, ..., a_in}, followed by
milestone M_{i+1}, edges are:

    M_i -> a_i1
    M_i -> a_i2
    ...
    a_i1 -> M_{i+1}
    a_i2 -> M_{i+1}
    ...

For the first and last milestone:

    START -> M_1
    a_last_j -> END

This yields a milestone-layered DAG that highlights temporal evolution across
milestones without inferring an order among attack steps inside the same
milestone.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MITRE_TACTIC_ORDER = [
    "reconnaissance",
    "resource-development",
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
]


@dataclass(frozen=True)
class MilestoneGroup:
    milestone_id: str
    tactic: str
    label: str
    order_key: tuple[int, int, str]
    attack_steps: list[dict[str, Any]]


def build_temporal_milestone_graph(
    *,
    run_id: str,
    case_id: str,
    attack_steps: list[dict[str, Any]],
    descriptions: list[dict[str, Any]] | None = None,
    final_conditions: list[dict[str, Any]] | None = None,
    retrieval_results: list[dict[str, Any]] | None = None,
    chunks: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    tactic_order: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build one temporal milestone DAG for a single case.

    Parameters
    ----------
    run_id:
        Experiment run identifier.
    case_id:
        Case identifier.
    attack_steps:
        Attack-step records, usually loaded from attack_steps.jsonl.
    descriptions:
        Generated description records, usually loaded from generated_descriptions.jsonl.
    final_conditions:
        Accepted/verified condition records, usually loaded from final_conditions.jsonl.
    retrieval_results:
        Retrieval records for attaching retrieved chunk IDs to step properties.
    chunks:
        Chunk records for optional evidence metadata.
    config:
        Run config.json content.
    tactic_order:
        Optional explicit tactic order. Defaults to MITRE ATT&CK Enterprise tactic order.

    Returns
    -------
    dict
        Serializable graph object with nodes, edges, and metadata.
    """
    descriptions = descriptions or []
    final_conditions = final_conditions or []
    retrieval_results = retrieval_results or []
    chunks = chunks or []
    config = config or {}
    tactic_order = normalize_tactic_order(tactic_order or load_tactic_order_from_config(config) or DEFAULT_MITRE_TACTIC_ORDER)

    case_steps = [step for step in attack_steps if str(step.get("case_id", case_id)) == case_id]
    if not case_steps:
        raise ValueError(f"No attack steps found for case_id={case_id!r}")

    description_by_step = _latest_by_step(descriptions, "attack_step_id")
    conditions_by_step = _conditions_by_step(final_conditions)
    retrieved_by_step = _retrieved_chunks_by_step(retrieval_results)
    chunk_meta_by_id = _chunk_meta_by_id(chunks)

    milestone_groups = group_attack_steps_into_milestones(
        case_id=case_id,
        attack_steps=case_steps,
        tactic_order=tactic_order,
    )
    if not milestone_groups:
        raise ValueError(f"No non-empty milestone groups found for case_id={case_id!r}")

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    start_node_id = f"start_{_slug(case_id)}"
    end_node_id = f"end_{_slug(case_id)}"

    nodes.append(
        {
            "node_id": start_node_id,
            "node_type": "start",
            "case_id": case_id,
            "label": "START",
        }
    )

    milestone_node_ids: list[str] = []
    attack_step_node_ids_by_milestone: dict[str, list[str]] = {}

    for milestone_order, group in enumerate(milestone_groups, start=1):
        if not group.attack_steps:
            # This invariant is central to the graph model: empty milestones are
            # not materialized.
            continue

        milestone_node_id = f"milestone_{_slug(group.milestone_id)}"
        milestone_node_ids.append(milestone_node_id)

        nodes.append(
            {
                "node_id": milestone_node_id,
                "node_type": "milestone",
                "case_id": case_id,
                "milestone_id": group.milestone_id,
                "label": group.label,
                "tactic": group.tactic,
                "milestone_order": milestone_order,
                "num_attack_steps": len(group.attack_steps),
            }
        )

        attack_step_node_ids: list[str] = []
        for step in group.attack_steps:
            attack_step_id = str(step.get("attack_step_id", ""))
            if not attack_step_id:
                raise ValueError(f"Attack step without attack_step_id in case_id={case_id!r}")

            step_node_id = f"step_{_slug(attack_step_id)}"
            attack_step_node_ids.append(step_node_id)

            step_description = description_by_step.get(attack_step_id)
            step_conditions = conditions_by_step.get(attack_step_id, {"pre": [], "post": []})
            retrieved_chunk_ids = retrieved_by_step.get(attack_step_id, [])
            cited_chunk_ids = _unique_preserve_order(
                _as_list(step_description.get("evidence_chunk_ids")) if step_description else []
                + _condition_evidence_ids(step_conditions["pre"])
                + _condition_evidence_ids(step_conditions["post"])
            )
            evidence_chunk_ids = _unique_preserve_order(cited_chunk_ids + retrieved_chunk_ids)
            source_documents = _source_documents_for_evidence(evidence_chunk_ids, chunk_meta_by_id)

            nodes.append(
                {
                    "node_id": step_node_id,
                    "node_type": "attack_step",
                    "case_id": case_id,
                    "attack_step_id": attack_step_id,
                    "label": _attack_step_label(step),
                    "milestone_id": group.milestone_id,
                    "attack_tactic": group.tactic,
                    "mitre_technique_id": step.get("attack_technique_id") or step.get("mitre_technique_id", ""),
                    "description": _description_property(step_description),
                    "pre_conditions": step_conditions["pre"],
                    "post_conditions": step_conditions["post"],
                    "evidence_chunk_ids": evidence_chunk_ids,
                    "cited_chunk_ids": cited_chunk_ids,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "source_documents": source_documents,
                    "source_order": _source_order(step),
                    "raw_attack_step": step,
                }
            )

        attack_step_node_ids_by_milestone[milestone_node_id] = attack_step_node_ids

    nodes.append(
        {
            "node_id": end_node_id,
            "node_type": "end",
            "case_id": case_id,
            "label": "END",
        }
    )

    # START -> first milestone
    edge_counter = 1
    edges.append(_edge(edge_counter, start_node_id, milestone_node_ids[0], "start_to_milestone"))
    edge_counter += 1

    # Milestone -> unordered attack-step set; attack-step set -> next milestone/END.
    for idx, milestone_node_id in enumerate(milestone_node_ids):
        step_node_ids = attack_step_node_ids_by_milestone[milestone_node_id]
        if not step_node_ids:
            raise ValueError(f"Milestone {milestone_node_id!r} has no attack steps")

        next_node_id = milestone_node_ids[idx + 1] if idx + 1 < len(milestone_node_ids) else end_node_id

        for step_node_id in step_node_ids:
            edges.append(_edge(edge_counter, milestone_node_id, step_node_id, "milestone_to_attack_step"))
            edge_counter += 1

        for step_node_id in step_node_ids:
            edges.append(_edge(edge_counter, step_node_id, next_node_id, "attack_step_to_next_milestone" if next_node_id != end_node_id else "attack_step_to_end"))
            edge_counter += 1

    graph = {
        "run_id": run_id,
        "case_id": case_id,
        "graph_type": "temporal_milestone_dag",
        "graph_semantics": {
            "start_node": "The unique entry point of the attack timeline.",
            "end_node": "The unique exit point of the attack timeline.",
            "milestone_node": "A tactic-aligned temporal phase containing at least one attack step.",
            "attack_step_node": "A STIX/ATT&CK-derived attack step enriched with descriptions, conditions, and evidence properties.",
            "within_milestone_order": "Attack steps inside the same milestone are not ordered unless an explicit ordering source is added externally.",
            "condition_representation": "Descriptions, pre-conditions, post-conditions, and evidence are attack-step properties, not graph nodes.",
            "verification": "No verifier labels are emitted; evidence links are deterministic RAG provenance.",
        },
        "nodes": nodes,
        "edges": edges,
        "metadata": _metadata(config, tactic_order),
    }

    graph["validation"] = validate_temporal_milestone_graph(graph)
    if not graph["validation"]["valid"]:
        messages = "; ".join(graph["validation"].get("errors", []))
        raise ValueError(f"Invalid temporal milestone graph for case_id={case_id!r}: {messages}")

    return graph


def group_attack_steps_into_milestones(
    *,
    case_id: str,
    attack_steps: list[dict[str, Any]],
    tactic_order: list[str],
) -> list[MilestoneGroup]:
    """
    Group attack steps into non-empty milestone groups.

    The grouping key is the explicit attack-step milestone_id when present;
    otherwise it is derived from case_id and attack_tactic. This supports both
    the current one-milestone-per-tactic representation and future pipelines
    where the same tactic can appear in multiple milestone occurrences.
    """
    order_index = {tactic: idx for idx, tactic in enumerate(normalize_tactic_order(tactic_order))}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_first_seen: dict[str, int] = {}

    for idx, step in enumerate(attack_steps):
        tactic = _normalize_tactic(str(step.get("attack_tactic") or step.get("tactic") or "other"))
        milestone_id = str(step.get("milestone_id") or f"{case_id}_{tactic}")
        groups[milestone_id].append(step)
        group_first_seen.setdefault(milestone_id, idx)

    milestone_groups: list[MilestoneGroup] = []
    for milestone_id, steps in groups.items():
        if not steps:
            continue
        tactic = _normalize_tactic(str(steps[0].get("attack_tactic") or steps[0].get("tactic") or _tactic_from_milestone_id(milestone_id)))
        label = _human_label(tactic)
        tactic_idx = order_index.get(tactic, len(order_index))
        first_seen = group_first_seen.get(milestone_id, 999999)
        milestone_groups.append(
            MilestoneGroup(
                milestone_id=milestone_id,
                tactic=tactic,
                label=label,
                order_key=(tactic_idx, first_seen, milestone_id),
                attack_steps=list(steps),
            )
        )

    return sorted(milestone_groups, key=lambda group: group.order_key)


def validate_temporal_milestone_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Validate core graph invariants."""
    errors: list[str] = []
    warnings: list[str] = []

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_by_id = {node.get("node_id"): node for node in nodes}

    start_nodes = [node for node in nodes if node.get("node_type") == "start"]
    end_nodes = [node for node in nodes if node.get("node_type") == "end"]
    milestone_nodes = [node for node in nodes if node.get("node_type") == "milestone"]
    attack_step_nodes = [node for node in nodes if node.get("node_type") == "attack_step"]

    if len(start_nodes) != 1:
        errors.append(f"Expected exactly one START node, found {len(start_nodes)}")
    if len(end_nodes) != 1:
        errors.append(f"Expected exactly one END node, found {len(end_nodes)}")
    if not milestone_nodes:
        errors.append("Expected at least one milestone node")
    if not attack_step_nodes:
        errors.append("Expected at least one attack-step node")

    for milestone in milestone_nodes:
        if int(milestone.get("num_attack_steps") or 0) < 1:
            errors.append(f"Milestone {milestone.get('node_id')} has no attack steps")

    milestone_ids = {node.get("milestone_id") for node in milestone_nodes}
    for step in attack_step_nodes:
        milestone_id = step.get("milestone_id")
        if milestone_id not in milestone_ids:
            errors.append(f"Attack step {step.get('node_id')} points to unknown milestone_id={milestone_id!r}")

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_by_id:
            errors.append(f"Edge {edge.get('edge_id')} has unknown source {source!r}")
        if target not in node_by_id:
            errors.append(f"Edge {edge.get('edge_id')} has unknown target {target!r}")
        if edge.get("label"):
            warnings.append(f"Edge {edge.get('edge_id')} has a visual label; labels should be suppressed by default")

    if _has_attack_step_to_attack_step_edges(edges, node_by_id):
        errors.append("Graph contains attack_step -> attack_step edges, which would imply unsupported within-milestone order")

    if not _is_acyclic(nodes, edges):
        errors.append("Graph is cyclic")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "invariants": {
            "exactly_one_start": len(start_nodes) == 1,
            "exactly_one_end": len(end_nodes) == 1,
            "milestones_non_empty": all(int(node.get("num_attack_steps") or 0) >= 1 for node in milestone_nodes),
            "attack_steps_have_one_milestone": not any(step.get("milestone_id") not in milestone_ids for step in attack_step_nodes),
            "no_attack_step_to_attack_step_edges": not _has_attack_step_to_attack_step_edges(edges, node_by_id),
            "acyclic": _is_acyclic(nodes, edges),
        },
    }


def normalize_tactic_order(tactic_order: list[str]) -> list[str]:
    return [_normalize_tactic(tactic) for tactic in tactic_order]


def load_tactic_order_from_file(path: str | Path) -> list[str]:
    """Load tactic order from a MITRE tactic JSON file when available."""
    path = Path(path)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    tactics: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                tactics.append(item)
            elif isinstance(item, dict):
                value = item.get("x_mitre_shortname") or item.get("shortname") or item.get("name") or item.get("tactic")
                if value:
                    tactics.append(str(value))
    elif isinstance(data, dict):
        for key in ["tactics", "objects", "items"]:
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        tactics.append(item)
                    elif isinstance(item, dict):
                        tactic = item.get("x_mitre_shortname") or item.get("shortname") or item.get("name") or item.get("tactic")
                        if tactic:
                            tactics.append(str(tactic))

    return normalize_tactic_order(tactics)


def load_tactic_order_from_config(config: dict[str, Any]) -> list[str]:
    graph_cfg = config.get("graph", {}) if isinstance(config, dict) else {}
    if isinstance(graph_cfg.get("tactic_order"), list):
        return normalize_tactic_order([str(x) for x in graph_cfg["tactic_order"]])
    tactic_file = graph_cfg.get("tactic_order_file") or "mitre/mitre-tactics.json"
    return load_tactic_order_from_file(tactic_file)


def graph_summary(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    milestone_nodes = [node for node in nodes if node.get("node_type") == "milestone"]
    attack_step_nodes = [node for node in nodes if node.get("node_type") == "attack_step"]

    steps_with_description = [node for node in attack_step_nodes if node.get("description", {}).get("text")]
    steps_with_pre = [node for node in attack_step_nodes if node.get("pre_conditions")]
    steps_with_post = [node for node in attack_step_nodes if node.get("post_conditions")]
    steps_with_conditions = [node for node in attack_step_nodes if node.get("pre_conditions") or node.get("post_conditions")]
    steps_with_evidence = [node for node in attack_step_nodes if node.get("evidence_chunk_ids")]

    condition_counts = [len(node.get("pre_conditions", [])) + len(node.get("post_conditions", [])) for node in attack_step_nodes]
    milestone_tactics = [str(node.get("tactic", "")) for node in milestone_nodes]

    return {
        "run_id": graph.get("run_id", ""),
        "case_id": graph.get("case_id", ""),
        "graph_type": graph.get("graph_type", ""),
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_milestones": len(milestone_nodes),
        "num_attack_steps": len(attack_step_nodes),
        "num_start_nodes": len([node for node in nodes if node.get("node_type") == "start"]),
        "num_end_nodes": len([node for node in nodes if node.get("node_type") == "end"]),
        "steps_with_description": len(steps_with_description),
        "steps_with_description_rate": _safe_divide(len(steps_with_description), len(attack_step_nodes)),
        "steps_with_preconditions": len(steps_with_pre),
        "steps_with_preconditions_rate": _safe_divide(len(steps_with_pre), len(attack_step_nodes)),
        "steps_with_postconditions": len(steps_with_post),
        "steps_with_postconditions_rate": _safe_divide(len(steps_with_post), len(attack_step_nodes)),
        "steps_with_conditions": len(steps_with_conditions),
        "steps_with_conditions_rate": _safe_divide(len(steps_with_conditions), len(attack_step_nodes)),
        "steps_with_evidence": len(steps_with_evidence),
        "steps_with_evidence_rate": _safe_divide(len(steps_with_evidence), len(attack_step_nodes)),
        "avg_conditions_per_step": _safe_divide(sum(condition_counts), len(condition_counts)),
        "max_conditions_per_step": max(condition_counts) if condition_counts else 0,
        "tactic_sequence": " > ".join(milestone_tactics),
        "graph_valid": graph.get("validation", {}).get("valid", False),
        "validation_errors": "; ".join(graph.get("validation", {}).get("errors", [])),
        "validation_warnings": "; ".join(graph.get("validation", {}).get("warnings", [])),
    }


def _metadata(config: dict[str, Any], tactic_order: list[str]) -> dict[str, Any]:
    retrieval = config.get("retrieval", {}) if isinstance(config, dict) else {}
    chunking = config.get("chunking", {}) if isinstance(config, dict) else {}
    generation = config.get("generation", {}) if isinstance(config, dict) else {}
    models = config.get("models", {}) if isinstance(config, dict) else {}
    return {
        "retriever": retrieval.get("strategy", ""),
        "retrieval_enabled": retrieval.get("enabled", None),
        "chunking_strategy": chunking.get("strategy", ""),
        "generation_enabled": generation.get("enabled", None),
        "verification_enabled": False,
        "text_generation_model": models.get("text_generation", ""),
        "embedding_backend": models.get("embedding_backend", ""),
        "embedding_model": models.get("sentence_transformer", models.get("embeddings", "")),
        "tactic_order": tactic_order,
    }


def _edge(edge_number: int, source: str, target: str, edge_type: str) -> dict[str, Any]:
    return {
        "edge_id": f"edge_{edge_number:05d}",
        "source": source,
        "target": target,
        "edge_type": edge_type,
    }


def _latest_by_step(records: list[dict[str, Any]], step_key: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        step_id = str(record.get(step_key, ""))
        if step_id:
            output[step_id] = record
    return output


def _conditions_by_step(conditions: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"pre": [], "post": []})
    for condition in conditions:
        step_id = str(condition.get("attack_step_id", ""))
        condition_type = str(condition.get("condition_type", "")).strip().lower()
        if not step_id or condition_type not in {"pre", "post"}:
            continue
        output[step_id][condition_type].append(_condition_property(condition))
    return dict(output)


def _condition_property(condition: dict[str, Any]) -> dict[str, Any]:
    """Return the graph property for a generated condition.

    The production pipeline does not emit verifier/support labels. Conditions
    therefore contain only stable IDs, type, text, and deterministic evidence.
    """
    return {
        "condition_id": condition.get("condition_id", ""),
        "condition_type": condition.get("condition_type", ""),
        "statement": condition.get("statement", ""),
        "evidence_chunk_ids": _as_list(condition.get("evidence_chunk_ids")),
        "source_documents": _as_list(condition.get("source_documents")),
    }


def _description_property(description: dict[str, Any] | None) -> dict[str, Any]:
    if not description:
        return {
            "text": "",
            "evidence_chunk_ids": [],
            "source_documents": [],
        }
    return {
        "text": description.get("description", ""),
        "evidence_chunk_ids": _as_list(description.get("evidence_chunk_ids")),
        "source_documents": _as_list(description.get("source_documents")),
    }


def _retrieved_chunks_by_step(retrieval_results: list[dict[str, Any]]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = defaultdict(list)
    for record in retrieval_results:
        step_id = str(record.get("attack_step_id", ""))
        if not step_id:
            continue
        for result in record.get("results", []):
            chunk_id = str(result.get("chunk_id", ""))
            if chunk_id:
                output[step_id].append(chunk_id)
    return {step_id: _unique_preserve_order(ids) for step_id, ids in output.items()}


def _chunk_meta_by_id(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        if chunk_id:
            output[chunk_id] = chunk
    return output


def _condition_evidence_ids(conditions: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for condition in conditions:
        ids.extend(_as_list(condition.get("evidence_chunk_ids")))
    return ids


def _source_documents_for_evidence(evidence_chunk_ids: list[str], chunk_meta_by_id: dict[str, dict[str, Any]]) -> list[str]:
    docs: list[str] = []
    for chunk_id in evidence_chunk_ids:
        chunk = chunk_meta_by_id.get(chunk_id, {})
        source = chunk.get("source_path") or chunk.get("source") or chunk.get("document_id")
        if source:
            docs.append(str(source))
    return _unique_preserve_order(docs)


def _attack_step_label(step: dict[str, Any]) -> str:
    for key in ["name", "original_label", "label", "attack_step", "input_text"]:
        value = str(step.get(key, "")).strip()
        if value:
            return value[:120]
    return str(step.get("attack_step_id", "attack_step"))


def _source_order(step: dict[str, Any]) -> int | None:
    for key in ["source_order", "step_index", "index"]:
        value = step.get(key)
        try:
            return int(value)
        except Exception:
            pass
    metadata = step.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ["source_order", "step_index", "index"]:
            value = metadata.get(key)
            try:
                return int(value)
            except Exception:
                pass
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    if isinstance(value, str):
        if ";" in value:
            return [part.strip() for part in value.split(";") if part.strip()]
        return [value] if value.strip() else []
    return [str(value)]


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _normalize_tactic(value: str) -> str:
    value = value.strip().lower().replace("_", "-").replace(" ", "-")
    value = re.sub(r"[^a-z0-9-]+", "", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "other"


def _tactic_from_milestone_id(milestone_id: str) -> str:
    parts = milestone_id.split("_", 1)
    if len(parts) == 2:
        return _normalize_tactic(parts[1])
    return _normalize_tactic(milestone_id)


def _human_label(tactic: str) -> str:
    return tactic.replace("-", " ").title()


def _slug(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "node"


def _has_attack_step_to_attack_step_edges(edges: list[dict[str, Any]], node_by_id: dict[str, dict[str, Any]]) -> bool:
    for edge in edges:
        source_type = node_by_id.get(edge.get("source"), {}).get("node_type")
        target_type = node_by_id.get(edge.get("target"), {}).get("node_type")
        if source_type == "attack_step" and target_type == "attack_step":
            return True
    return False


def _is_acyclic(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> bool:
    adjacency: dict[str, list[str]] = defaultdict(list)
    node_ids = {str(node.get("node_id")) for node in nodes}
    for edge in edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source in node_ids and target in node_ids:
            adjacency[source].append(target)

    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node_id: str) -> bool:
        if node_id in visiting:
            return False
        if node_id in visited:
            return True
        visiting.add(node_id)
        for neighbor in adjacency.get(node_id, []):
            if not dfs(neighbor):
                return False
        visiting.remove(node_id)
        visited.add(node_id)
        return True

    return all(dfs(node_id) for node_id in node_ids)


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
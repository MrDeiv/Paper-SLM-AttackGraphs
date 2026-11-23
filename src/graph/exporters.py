"""
Exporters for temporal milestone attack graphs.

The JSON export preserves all graph attributes. DOT and GraphML are intended for
visualization and interoperability. DOT deliberately suppresses edge labels so
that the rendered graph emphasizes the temporal/milestone structure rather than
repeating identical labels on every edge.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


MILESTONE_SHAPE = "box"
ATTACK_STEP_SHAPE = "ellipse"
TERMINAL_SHAPE = "oval"


def export_graph_json(graph: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")


def export_graph_dot(graph: dict[str, Any], path: str | Path, *, rankdir: str = "LR") -> None:
    """Export a Graphviz DOT file with unlabeled edges."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    graph_name = _dot_id(f"{graph.get('case_id', 'attack')}_temporal_milestone_dag")
    lines.append(f"digraph {graph_name} {{")
    lines.append(f"  rankdir={rankdir};")
    lines.append("  graph [fontname=\"Helvetica\", fontsize=10];")
    lines.append("  node [fontname=\"Helvetica\", fontsize=10];")
    lines.append("  edge [fontname=\"Helvetica\", fontsize=9, arrowsize=0.8];")
    lines.append("")

    for node in graph.get("nodes", []):
        node_id = str(node.get("node_id", ""))
        if not node_id:
            continue
        attrs = _dot_node_attrs(node)
        attr_text = ", ".join(f"{key}=\"{_escape_dot(value)}\"" for key, value in attrs.items())
        lines.append(f"  \"{_escape_dot(node_id)}\" [{attr_text}];")

    lines.append("")
    for edge in graph.get("edges", []):
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source and target:
            lines.append(f"  \"{_escape_dot(source)}\" -> \"{_escape_dot(target)}\";")

    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_graph_graphml(graph: dict[str, Any], path: str | Path) -> None:
    """
    Export GraphML with selected scalar node/edge attributes.

    Complex properties such as conditions are serialized as JSON strings on the
    attack-step node. Edge labels are not emitted.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    node_keys = {
        "node_type": "string",
        "case_id": "string",
        "label": "string",
        "milestone_id": "string",
        "tactic": "string",
        "milestone_order": "int",
        "num_attack_steps": "int",
        "attack_step_id": "string",
        "attack_tactic": "string",
        "mitre_technique_id": "string",
        "description_text": "string",
        "pre_conditions_json": "string",
        "post_conditions_json": "string",
        "evidence_chunk_ids": "string",
        "source_documents": "string",
    }
    edge_keys = {
        "edge_type": "string",
    }

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')

    key_id_by_name: dict[str, str] = {}
    key_counter = 0
    for name, attr_type in node_keys.items():
        key_counter += 1
        key_id = f"d{key_counter}"
        key_id_by_name[f"node:{name}"] = key_id
        lines.append(f'  <key id="{key_id}" for="node" attr.name="{html.escape(name)}" attr.type="{attr_type}"/>')
    for name, attr_type in edge_keys.items():
        key_counter += 1
        key_id = f"d{key_counter}"
        key_id_by_name[f"edge:{name}"] = key_id
        lines.append(f'  <key id="{key_id}" for="edge" attr.name="{html.escape(name)}" attr.type="{attr_type}"/>')

    graph_id = html.escape(str(graph.get("case_id", "attack_graph")))
    lines.append(f'  <graph id="{graph_id}" edgedefault="directed">')

    for node in graph.get("nodes", []):
        node_id = str(node.get("node_id", ""))
        if not node_id:
            continue
        flat = _flatten_node_for_graphml(node)
        lines.append(f'    <node id="{html.escape(node_id)}">')
        for name in node_keys:
            value = flat.get(name)
            if value in (None, ""):
                continue
            key_id = key_id_by_name[f"node:{name}"]
            lines.append(f'      <data key="{key_id}">{html.escape(str(value))}</data>')
        lines.append("    </node>")

    for edge in graph.get("edges", []):
        edge_id = str(edge.get("edge_id", ""))
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if not edge_id or not source or not target:
            continue
        lines.append(f'    <edge id="{html.escape(edge_id)}" source="{html.escape(source)}" target="{html.escape(target)}">')
        edge_type = edge.get("edge_type")
        if edge_type:
            key_id = key_id_by_name["edge:edge_type"]
            lines.append(f'      <data key="{key_id}">{html.escape(str(edge_type))}</data>')
        lines.append("    </edge>")

    lines.append("  </graph>")
    lines.append("</graphml>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_all_graph_formats(graph: dict[str, Any], output_prefix: str | Path) -> dict[str, str]:
    """
    Export JSON, DOT, and GraphML using output_prefix without extension.
    """
    output_prefix = Path(output_prefix)
    paths = {
        "json": str(output_prefix.with_suffix(".json")),
        "dot": str(output_prefix.with_suffix(".dot")),
        "graphml": str(output_prefix.with_suffix(".graphml")),
    }
    export_graph_json(graph, paths["json"])
    export_graph_dot(graph, paths["dot"])
    export_graph_graphml(graph, paths["graphml"])
    return paths


def _dot_node_attrs(node: dict[str, Any]) -> dict[str, str]:
    node_type = str(node.get("node_type", ""))
    label = str(node.get("label", node.get("node_id", "")))

    if node_type == "milestone":
        return {
            "label": label,
            "shape": MILESTONE_SHAPE,
            "style": "rounded,bold",
        }
    if node_type == "attack_step":
        technique = str(node.get("mitre_technique_id", "")).strip()
        display = label if not technique else f"{label}\n{technique}"
        return {
            "label": display,
            "shape": ATTACK_STEP_SHAPE,
        }
    if node_type in {"start", "end"}:
        return {
            "label": label,
            "shape": TERMINAL_SHAPE,
            "style": "bold",
        }
    return {
        "label": label,
        "shape": "ellipse",
    }


def _flatten_node_for_graphml(node: dict[str, Any]) -> dict[str, Any]:
    description = node.get("description", {}) if isinstance(node.get("description"), dict) else {}
    return {
        "node_type": node.get("node_type", ""),
        "case_id": node.get("case_id", ""),
        "label": node.get("label", ""),
        "milestone_id": node.get("milestone_id", ""),
        "tactic": node.get("tactic", ""),
        "milestone_order": node.get("milestone_order", ""),
        "num_attack_steps": node.get("num_attack_steps", ""),
        "attack_step_id": node.get("attack_step_id", ""),
        "attack_tactic": node.get("attack_tactic", ""),
        "mitre_technique_id": node.get("mitre_technique_id", ""),
        "description_text": description.get("text", ""),
        "pre_conditions_json": json.dumps(node.get("pre_conditions", []), ensure_ascii=False),
        "post_conditions_json": json.dumps(node.get("post_conditions", []), ensure_ascii=False),
        "evidence_chunk_ids": ";".join(str(x) for x in node.get("evidence_chunk_ids", [])),
        "source_documents": ";".join(str(x) for x in node.get("source_documents", [])),
    }


def _dot_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))
    return safe or "attack_graph"


def _escape_dot(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
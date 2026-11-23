#!/usr/bin/env python3
"""
Build temporal milestone attack graphs from experiment run outputs.

The script is deterministic and does not call any LLM. It assembles graph
artifacts from existing JSONL files:

- attack_steps.jsonl
- generated_descriptions.jsonl
- final_conditions.jsonl
- retrieval_results.jsonl
- chunks.jsonl
- config.json

Outputs per case:

- graphs/<case_id>_temporal_milestone_graph.json
- graphs/<case_id>_temporal_milestone_graph.dot
- graphs/<case_id>_temporal_milestone_graph.graphml

Example:
    python scripts/build_attack_graphs.py --runs outputs/pilot_runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graph.exporters import export_all_graph_formats
from src.graph.temporal_milestone_graph import build_temporal_milestone_graph, graph_summary, load_tactic_order_from_file
from src.io_utils import read_jsonl
import csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal milestone attack graphs from run outputs.")
    parser.add_argument("--runs", default="outputs/runs", help="Directory containing run output folders.")
    parser.add_argument("--run", action="append", help="Specific run_id to process. Can be repeated.")
    parser.add_argument("--case", action="append", help="Specific case_id to process. Can be repeated.")
    parser.add_argument("--tactic-order-file", default="mitre/mitre-tactics.json", help="Optional MITRE tactic-order JSON file.")
    parser.add_argument("--out-subdir", default="graphs", help="Subdirectory inside each run folder for graph outputs.")
    parser.add_argument("--summary-out", default=None, help="Optional CSV path for graph summaries.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with code 1 if any graph fails to build.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = _discover_run_dirs(Path(args.runs), args.run)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {args.runs}")

    tactic_order = load_tactic_order_from_file(args.tactic_order_file)
    selected_cases = set(args.case or [])
    all_summary_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for run_dir in run_dirs:
        try:
            summary_rows = build_graphs_for_run(
                run_dir=run_dir,
                selected_cases=selected_cases,
                tactic_order=tactic_order,
                out_subdir=args.out_subdir,
            )
            all_summary_rows.extend(summary_rows)
            print(f"Built graphs for {run_dir.name}: {len(summary_rows)} case(s)")
        except Exception as exc:
            message = f"{run_dir.name}: {exc!r}"
            errors.append(message)
            print(f"ERROR {message}")
            if args.fail_on_error:
                raise

    summary_out = Path(args.summary_out) if args.summary_out else Path(args.runs) / "temporal_milestone_graph_summary.csv"
    _write_csv(summary_out, all_summary_rows)
    print(f"Graph summary rows: {len(all_summary_rows)}")
    print(f"Summary: {summary_out}")

    if errors and args.fail_on_error:
        raise SystemExit(1)


def build_graphs_for_run(
    *,
    run_dir: Path,
    selected_cases: set[str],
    tactic_order: list[str],
    out_subdir: str,
) -> list[dict[str, Any]]:
    config = _read_json(run_dir / "config.json")
    run_id = str(config.get("run", {}).get("run_id", run_dir.name))

    attack_steps = read_jsonl(run_dir / "attack_steps.jsonl")
    descriptions = read_jsonl(run_dir / "generated_descriptions.jsonl")
    final_conditions = read_jsonl(run_dir / "final_conditions.jsonl")
    retrieval_results = read_jsonl(run_dir / "retrieval_results.jsonl")
    chunks = read_jsonl(run_dir / "chunks.jsonl")

    case_ids = sorted({str(step.get("case_id", "")) for step in attack_steps if step.get("case_id")})
    if selected_cases:
        case_ids = [case_id for case_id in case_ids if case_id in selected_cases]

    graph_dir = run_dir / out_subdir
    graph_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for case_id in case_ids:
        graph = build_temporal_milestone_graph(
            run_id=run_id,
            case_id=case_id,
            attack_steps=attack_steps,
            descriptions=[row for row in descriptions if str(row.get("case_id", case_id)) == case_id],
            final_conditions=[row for row in final_conditions if str(row.get("case_id", case_id)) == case_id],
            retrieval_results=[row for row in retrieval_results if str(row.get("case_id", case_id)) == case_id],
            chunks=[row for row in chunks if str(row.get("case_id", case_id)) == case_id],
            config=config,
            tactic_order=tactic_order,
        )
        prefix = graph_dir / f"{case_id}_temporal_milestone_graph"
        paths = export_all_graph_formats(graph, prefix)
        summary = graph_summary(graph)
        summary.update(
            {
                "run_dir": str(run_dir),
                "json_path": paths["json"],
                "dot_path": paths["dot"],
                "graphml_path": paths["graphml"],
            }
        )
        summaries.append(summary)

    return summaries


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _discover_run_dirs(runs_root: Path, selected_runs: list[str] | None) -> list[Path]:
    if selected_runs:
        return [runs_root / run_id for run_id in selected_runs if (runs_root / run_id).exists()]
    return sorted(path for path in runs_root.iterdir() if path.is_dir() and (path / "config.json").exists())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


if __name__ == "__main__":
    main()
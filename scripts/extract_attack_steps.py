#!/usr/bin/env python3
"""Extract attack-step identifiers for one configured case.

Purpose
-------
This utility is meant to be used before running fine-grained evaluation with:

    python scripts/evaluate_case.py --case <case_id> --as <attack_step_id>

It reads the same final-evaluation configuration used by evaluate_case.py, loads
attack steps using the existing production STIX extraction component, and writes a
small index file containing the stable attack_step_id values.

The script does not run RAG, generation, verification, or metrics.  It only
extracts/list attack-step metadata.

Examples
--------
List the attack steps for one case and write the default CSV:

    python scripts/extract_attack_steps.py --case goofy

Print a markdown table to the terminal:

    python scripts/extract_attack_steps.py --case goofy --format markdown --stdout

Force extraction from STIX, ignoring any previously persisted attack_steps.jsonl:

    python scripts/extract_attack_steps.py --case goofy --source stix

Also persist the production attack-step JSONL in case_outputs/<case_id>/:

    python scripts/extract_attack_steps.py --case goofy --persist-jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

# Make repository-root imports work when the script is executed as:
#   python scripts/extract_attack_steps.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final_eval_common import (  # noqa: E402
    artifact_path,
    case_output_dir,
    evaluation_case_dir,
    get_artifact_names,
    get_case_entry,
    get_eval_cfg,
    load_config,
    read_jsonl,
    repo_path,
    stable_case_name,
    write_jsonl,
)
from src.attack_steps import load_attack_steps_from_stix  # noqa: E402


OUTPUT_COLUMNS = [
    "case_id",
    "case_name",
    "step_index",
    "attack_step_id",
    "milestone_id",
    "tactic_name",
    "technique_id",
    "technique_name",
    "attack_step_label",
    "original_description",
    "source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract attack-step IDs for one case, for use with evaluate_case.py --as."
    )
    parser.add_argument("--case", required=True, help="Case identifier, or 'all' to extract all configured cases.")
    parser.add_argument(
        "--config",
        default="configs/evaluation/final_pipeline_eval.yaml",
        help="Final evaluation config. Default: configs/evaluation/final_pipeline_eval.yaml",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "stix", "persisted"],
        default="auto",
        help=(
            "Source for attack steps. 'auto' uses case_outputs/<case>/attack_steps.jsonl when present, "
            "otherwise extracts from STIX. 'stix' always extracts from STIX. 'persisted' requires the "
            "existing case_outputs JSONL file. Default: auto."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["csv", "jsonl", "markdown"],
        default="csv",
        help="Output format. Default: csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path. If omitted, the script writes to evaluation_outputs/<case>/attack_steps_index.<ext> "
            "or evaluation_outputs/attack_steps_index_all.<ext> for --case all."
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the extracted attack-step index to stdout.",
    )
    parser.add_argument(
        "--persist-jsonl",
        action="store_true",
        help=(
            "Also write the raw production attack-step records to the configured case output artifact "
            "case_outputs/<case>/attack_steps.jsonl. This is useful before running evaluate_case.py --as."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    case_ids = resolve_case_ids(config, args.case)
    all_rows: list[dict[str, Any]] = []

    for case_id in case_ids:
        case_entry = get_case_entry(config, case_id)
        steps = load_steps(config, case_entry, source=args.source)
        validate_unique_step_ids(case_id, steps)

        if args.persist_jsonl:
            path = artifact_path(config, case_id, "attack_steps")
            write_jsonl(path, steps)
            print(f"[{case_id}] Persisted {len(steps)} attack steps to {path}")

        rows = [make_output_row(config, case_entry, step) for step in steps]
        all_rows.extend(rows)

    output_path = resolve_output_path(config, args.case, args.format, args.output)
    write_rows(output_path, all_rows, args.format)
    print(f"Wrote {len(all_rows)} attack-step rows to {output_path}")

    if args.stdout:
        if args.format == "markdown":
            print(markdown_table(all_rows))
        else:
            # Markdown is the most readable terminal representation even when
            # the file output is CSV/JSONL.
            print(markdown_table(all_rows))


def resolve_case_ids(config: dict[str, Any], requested: str) -> list[str]:
    if requested.lower() != "all":
        # get_case_entry performs validation and gives a useful error message.
        get_case_entry(config, requested)
        return [requested]

    manifest_path = get_eval_cfg(config).get("case_manifest", "data/case_manifest.json")
    manifest_file = repo_path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Case manifest not found: {manifest_file}")
    with manifest_file.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, list):
        raise ValueError(f"Case manifest must contain a list of case entries: {manifest_file}")
    return [str(entry["case_id"]) for entry in manifest]


def load_steps(config: dict[str, Any], case_entry: dict[str, Any], *, source: str) -> list[Any]:
    """Load attack steps either from persisted JSONL or from the configured STIX file."""
    case_id = str(case_entry["case_id"])
    persisted_path = artifact_path(config, case_id, "attack_steps")

    if source in {"auto", "persisted"} and persisted_path.exists():
        rows = read_jsonl(persisted_path)
        if rows:
            print(f"[{case_id}] Loaded {len(rows)} persisted attack steps from {persisted_path}")
            return rows

    if source == "persisted":
        raise FileNotFoundError(
            f"No persisted attack-step file found for case {case_id}: {persisted_path}. "
            "Use --source stix or --source auto to extract from STIX."
        )

    stix_file = repo_path(case_entry["stix_file"])
    mitre_tactics_file = str(case_entry.get("mitre_tactics_file") or "mitre/mitre-tactics.json")
    print(f"[{case_id}] Extracting attack steps from STIX: {stix_file}")
    return load_attack_steps_from_stix(
        str(stix_file),
        case_id=case_id,
        mitre_tactics_file=mitre_tactics_file,
    )


def make_output_row(config: dict[str, Any], case_entry: dict[str, Any], step: Any) -> dict[str, Any]:
    """Convert one production attack-step record into a flat index row.

    The row is intentionally compact: its main purpose is to expose the stable
    attack_step_id values that can be passed to:

        python scripts/evaluate_case.py --case <case_id> --as <attack_step_id>
    """
    record = to_dict(step)

    metadata = dict(record.get("metadata") or {})
    raw_action = metadata.get("raw_action") if isinstance(metadata.get("raw_action"), dict) else {}

    case_id = str(case_entry["case_id"])

    # Prefer an explicit human-readable case name from the manifest/config.
    # If not available, derive one deterministically from the case_id.
    case_name = (
        case_entry.get("case_name")
        or case_entry.get("name")
        or case_entry.get("title")
        or stable_case_name(case_id)
    )

    step_index = metadata.get("step_index")
    if step_index is None and isinstance(raw_action, dict):
        step_index = raw_action.get("step_index")

    technique_name = ""
    if isinstance(raw_action, dict):
        technique_name = raw_action.get("name", "") or raw_action.get("technique_name", "")
    if not technique_name:
        technique_name = record.get("technique_name", "") or record.get("original_label", "")

    return {
        "case_id": case_id,
        "case_name": case_name,
        "step_index": step_index if step_index is not None else "",
        "attack_step_id": record.get("attack_step_id", ""),
        "milestone_id": record.get("milestone_id", ""),
        "tactic_name": record.get("attack_tactic", "") or record.get("tactic_name", ""),
        "technique_id": record.get("attack_technique_id", "") or record.get("technique_id", ""),
        "technique_name": technique_name,
        "attack_step_label": record.get("original_label", "") or record.get("attack_step_label", ""),
        "original_description": squash_whitespace(record.get("input_text", "") or record.get("original_description", "")),
        "source": record.get("source", ""),
    }


def to_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    # Fallback for object-like records.
    out: dict[str, Any] = {}
    for key in [
        "attack_step_id",
        "case_id",
        "source",
        "original_label",
        "attack_tactic",
        "input_text",
        "attack_technique_id",
        "milestone_id",
        "stix_objects",
        "metadata",
    ]:
        if hasattr(value, key):
            out[key] = getattr(value, key)
    return out


def validate_unique_step_ids(case_id: str, steps: Iterable[Any]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for step in steps:
        step_id = str(to_dict(step).get("attack_step_id", ""))
        if not step_id:
            raise ValueError(f"Case {case_id} contains an attack step without attack_step_id")
        if step_id in seen:
            duplicates.add(step_id)
        seen.add(step_id)
    if duplicates:
        raise ValueError(f"Case {case_id} contains duplicate attack_step_id values: {sorted(duplicates)}")


def resolve_output_path(config: dict[str, Any], case_arg: str, output_format: str, explicit_output: str | None) -> Path:
    if explicit_output:
        return repo_path(explicit_output)

    suffix = "md" if output_format == "markdown" else output_format

    if case_arg.lower() == "all":
        root = repo_path(get_eval_cfg(config).get("evaluation_output_root", "evaluation_outputs"))
        root.mkdir(parents=True, exist_ok=True)
        return root / f"attack_steps_index_all.{suffix}"

    out_dir = evaluation_case_dir(config, case_arg)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"attack_steps_index.{suffix}"


def write_rows(path: Path, rows: list[dict[str, Any]], output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "csv":
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: csv_value(row.get(key, "")) for key in OUTPUT_COLUMNS})
        return

    if output_format == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    if output_format == "markdown":
        path.write_text(markdown_table(rows), encoding="utf-8")
        return

    raise ValueError(f"Unsupported output format: {output_format}")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """Return a compact markdown table focused on IDs needed by --as."""
    columns = [
        "step_index",
        "case_id",
        "attack_step_id",
        "tactic_name",
        "technique_id",
        "attack_step_label",
    ]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        values = [escape_md(str(row.get(col, ""))) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ||| ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def squash_whitespace(text: Any) -> str:
    return " ".join(str(text or "").split())


def escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()

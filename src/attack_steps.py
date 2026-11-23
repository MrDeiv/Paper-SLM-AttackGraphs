"""
Attack-step extraction helpers for experiment runners
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.STIXParser import STIXParser
from src.group_attack_patterns import group_attack_patterns
from src.id_utils import make_attack_step_id
from src.schemas import AttackStepRecord


DEFAULT_MITRE_TACTICS_FILE = "mitre/mitre-tactics.json"


def load_attack_steps_from_stix(
    stix_file: str,
    *,
    case_id: str,
    mitre_tactics_file: str = DEFAULT_MITRE_TACTICS_FILE,
) -> list[AttackStepRecord]:
    """
    Load STIX attack-pattern uses and convert them into ordered attack steps.
    """
    if not Path(stix_file).exists():
        raise FileNotFoundError(f"STIX file not found: {stix_file}")
    if not Path(mitre_tactics_file).exists():
        raise FileNotFoundError(f"MITRE tactics file not found: {mitre_tactics_file}")

    parser = STIXParser()
    parser.parse(stix_file)
    attack_patterns_used = parser.get_attack_pattern_used()
    mitre_tactics = json.loads(Path(mitre_tactics_file).read_text(encoding="utf-8"))
    grouped = group_attack_patterns(mitre_tactics, attack_patterns_used)

    records: list[AttackStepRecord] = []
    global_index = 1
    for tactic, actions in grouped.items():
        for action in actions:
            name = str(action.get("name", "Unnamed attack step"))
            description = str(action.get("description", ""))
            attack_step_id = make_attack_step_id(case_id, tactic, global_index, name)
            records.append(
                AttackStepRecord(
                    attack_step_id=attack_step_id,
                    case_id=case_id,
                    source="stix_attack_pattern",
                    original_label=name,
                    attack_tactic=tactic,
                    input_text=description,
                    attack_technique_id=_extract_attack_id(action),
                    milestone_id=f"{case_id}_{tactic}",
                    stix_objects=[],
                    metadata={
                        "raw_action": _json_safe(action),
                        "step_index": global_index,
                    },
                )
            )
            global_index += 1

    return records


def attack_step_query(step: AttackStepRecord) -> str:
    """
    Build a retrieval query for one attack step.
    """
    parts = [
        f"Attack tactic: {step.attack_tactic}",
        f"Attack step: {step.original_label}",
        f"Description: {step.input_text}",
    ]
    if step.attack_technique_id:
        parts.append(f"MITRE ATT&CK technique: {step.attack_technique_id}")
    return "\n".join(parts)


def _extract_attack_id(action: dict[str, Any]) -> str | None:
    refs = action.get("external_references") or []
    for ref in refs:
        external_id = ref.get("external_id") if isinstance(ref, dict) else None
        if external_id:
            return str(external_id)
    return None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)
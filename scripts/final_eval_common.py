"""Shared helpers for the final CTI pipeline evaluation scripts.

The functions in this module are intentionally lightweight and dependency-poor.
They centralize config loading, stable serialization, missing-value handling,
annotation-column definitions, and common validation so that the main scripts
remain readable and consistent with the evaluation protocol.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is listed in the project env.
    yaml = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    """Resolve project-relative paths without depending on the current cwd."""
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON configuration file."""
    path = repo_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML config files.")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be an object: {path}")
    return data


def load_json(path: str | Path) -> Any:
    with repo_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(data), f, indent=indent, ensure_ascii=False)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = repo_path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
            else:
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Any], *, append: bool = False) -> None:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = repo_path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path = repo_path(path)
    rows = [jsonable(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def jsonable(value: Any) -> Any:
    """Convert dataclasses and Paths to JSON/CSV-friendly values."""
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(jsonable(v) for v in value)
    return value


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ";".join(str(x) for x in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def get_eval_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("evaluation", {}))


def get_artifact_names(config: dict[str, Any]) -> dict[str, str]:
    return dict(config.get("artifacts", {}))


def case_output_dir(config: dict[str, Any], case_id: str) -> Path:
    return repo_path(get_eval_cfg(config).get("case_output_root", "case_outputs")) / case_id


def evaluation_case_dir(config: dict[str, Any], case_id: str) -> Path:
    return repo_path(get_eval_cfg(config).get("evaluation_output_root", "evaluation_outputs")) / case_id


def final_metrics_dir(config: dict[str, Any]) -> Path:
    return repo_path(get_eval_cfg(config).get("final_metrics_output_dir", "evaluation_final_metrics"))


def artifact_path(config: dict[str, Any], case_id: str, artifact_key: str, *, evaluation: bool = False) -> Path:
    names = get_artifact_names(config)
    filename = names.get(artifact_key, artifact_key)
    root = evaluation_case_dir(config, case_id) if evaluation else case_output_dir(config, case_id)
    return root / filename


def load_case_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = get_eval_cfg(config).get("case_manifest", "data/case_manifest.json")
    manifest = load_json(manifest_path)
    if not isinstance(manifest, list):
        raise ValueError("Case manifest must contain a list of case entries")
    return [dict(entry) for entry in manifest]


def get_case_entry(config: dict[str, Any], case_id: str) -> dict[str, Any]:
    for entry in load_case_manifest(config):
        if str(entry.get("case_id")) == case_id:
            return dict(entry)
    available = [str(entry.get("case_id")) for entry in load_case_manifest(config)]
    raise ValueError(f"Unknown case {case_id!r}. Available cases: {available}")


def annotation_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("annotation", {}))


def delimiter(config: dict[str, Any]) -> str:
    return str(annotation_cfg(config).get("delimiter", " ||| "))


def na_value(config: dict[str, Any]) -> str:
    return str(annotation_cfg(config).get("na_value", "NA"))


def annotation_columns(config: dict[str, Any]) -> list[str]:
    return [str(x) for x in annotation_cfg(config).get("columns", [])]


def condition_criteria(config: dict[str, Any]) -> list[str]:
    return [str(x) for x in annotation_cfg(config).get("condition_criteria", [])]


def description_criteria(config: dict[str, Any]) -> list[str]:
    return [str(x) for x in annotation_cfg(config).get("description_criteria", [])]


def all_criteria(config: dict[str, Any]) -> list[str]:
    return condition_criteria(config) + description_criteria(config)


def allowed_labels(config: dict[str, Any], criterion: str) -> set[str]:
    allowed = annotation_cfg(config).get("allowed_labels", {}).get(criterion, [])
    return {str(x) for x in allowed}


def missing_tokens(config: dict[str, Any]) -> set[str]:
    values = config.get("metrics", {}).get("missing_tokens", ["", "NA", "N/A", "none", "-"])
    return {str(x).strip().lower() for x in values}


def is_missing(value: Any, config: dict[str, Any]) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in missing_tokens(config)


def normalize_missing(value: Any, config: dict[str, Any]) -> str:
    return na_value(config) if is_missing(value, config) else str(value).strip()


def join_values(values: Iterable[Any], config: dict[str, Any]) -> str:
    clean = [str(v).strip() for v in values if str(v).strip() and str(v).strip() != na_value(config)]
    return delimiter(config).join(clean)


def split_values(value: Any, config: dict[str, Any]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text or text == na_value(config):
        return []
    if delimiter(config) in text:
        return [part.strip() for part in text.split(delimiter(config)) if part.strip()]
    # Be permissive when reading older files; writing always uses delimiter().
    return [part.strip() for part in re.split(r"\s*;\s*|\s*,\s*", text) if part.strip()]


def count_tokens(text: Any) -> int:
    """Whitespace tokenizer used when no model tokenizer is configured."""
    return len(str(text or "").split())


def safe_divide(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    return None if not vals else sum(vals) / len(vals)


def stable_case_name(case_id: str) -> str:
    """Human-readable fallback used when no explicit case_name is available."""
    return case_id.replace("_", " ").replace("-", " ").title()


def validate_annotation_units(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[str]:
    """Validate protocol-level constraints before writing annotation CSVs."""
    warnings: list[str] = []
    required = annotation_columns(config)
    if not required:
        raise ValueError("annotation.columns is empty in the config")

    seen: set[str] = set()
    for i, row in enumerate(rows, start=2):  # CSV line after header
        missing = [col for col in required if col not in row]
        if missing:
            raise ValueError(f"Annotation row {i} is missing columns: {missing}")

        unit_id = str(row.get("annotation_unit_id", "")).strip()
        if not unit_id:
            raise ValueError(f"Annotation row {i} has empty annotation_unit_id")
        if unit_id in seen:
            raise ValueError(f"Duplicate annotation_unit_id: {unit_id}")
        seen.add(unit_id)

        field_type = str(row.get("field_type", "")).strip()
        condition_type = str(row.get("condition_type", "")).strip()
        generated = str(row.get("generated_field_text", "")).strip()
        if not generated:
            raise ValueError(f"Annotation row {unit_id} has empty generated_field_text")

        if field_type == "condition" and condition_type not in {"precondition", "postcondition"}:
            raise ValueError(f"Condition row {unit_id} has invalid condition_type={condition_type!r}")
        if field_type == "description" and condition_type not in {"", na_value(config)}:
            raise ValueError(f"Description row {unit_id} must have empty/NA condition_type")

        evidence_ids = split_values(row.get("evidence_chunk_ids", ""), config)
        evidence_texts = split_values(row.get("evidence_texts", ""), config)
        if evidence_ids and len(evidence_texts) < len(evidence_ids):
            warnings.append(f"{unit_id}: fewer evidence texts than evidence chunk IDs")

    return warnings


def validate_same_annotation_ids(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]]) -> None:
    a_ids = {str(row.get("annotation_unit_id", "")).strip() for row in a_rows}
    b_ids = {str(row.get("annotation_unit_id", "")).strip() for row in b_rows}
    if a_ids != b_ids:
        only_a = sorted(a_ids - b_ids)[:10]
        only_b = sorted(b_ids - a_ids)[:10]
        raise ValueError(f"Annotator files have different annotation_unit_id sets. only_a={only_a}, only_b={only_b}")


def validate_labels(rows: list[dict[str, Any]], config: dict[str, Any], *, allow_missing: bool = False) -> None:
    """Validate filled annotation labels for applicable row types."""
    for row in rows:
        field_type = str(row.get("field_type", "")).strip()
        unit_id = str(row.get("annotation_unit_id", "")).strip()
        criteria = condition_criteria(config) if field_type == "condition" else description_criteria(config)
        for criterion in criteria:
            value = str(row.get(criterion, "")).strip()
            if is_missing(value, config):
                if allow_missing:
                    continue
                raise ValueError(f"Missing label for {criterion} in {unit_id}")
            if value not in allowed_labels(config, criterion):
                raise ValueError(
                    f"Invalid label for {criterion} in {unit_id}: {value!r}. "
                    f"Allowed: {sorted(allowed_labels(config, criterion))}"
                )


def raw_agreement(a: list[str], b: list[str]) -> float | None:
    if not a:
        return None
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohen_kappa(a: list[str], b: list[str], *, weights: str | None = None) -> float | None:
    """Compute Cohen's kappa, using sklearn when available and a fallback otherwise."""
    if not a or len(a) != len(b):
        return None
    try:
        from sklearn.metrics import cohen_kappa_score

        return float(cohen_kappa_score(a, b, weights=weights))
    except Exception:
        if weights:
            return weighted_kappa_fallback(a, b)
        return cohen_kappa_fallback(a, b)


def cohen_kappa_fallback(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = Counter(a)
    cb = Counter(b)
    labels = set(ca) | set(cb)
    expected = sum((ca[label] / n) * (cb[label] / n) for label in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def weighted_kappa_fallback(a: list[str], b: list[str]) -> float | None:
    """Quadratic weighted kappa fallback for ordinal integer labels."""
    labels = sorted({int(x) for x in a} | {int(x) for x in b})
    if not labels:
        return None
    if len(labels) == 1:
        return 1.0
    index = {label: i for i, label in enumerate(labels)}
    n = len(a)
    k = len(labels)

    observed = [[0.0 for _ in range(k)] for _ in range(k)]
    for x, y in zip(a, b):
        observed[index[int(x)]][index[int(y)]] += 1.0

    row_marginals = [sum(row) for row in observed]
    col_marginals = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    weighted_observed = 0.0
    weighted_expected = 0.0
    for i in range(k):
        for j in range(k):
            weight = ((i - j) ** 2) / ((k - 1) ** 2)
            expected = row_marginals[i] * col_marginals[j] / n
            weighted_observed += weight * observed[i][j]
            weighted_expected += weight * expected

    if weighted_expected == 0:
        return 1.0 if weighted_observed == 0 else None
    return 1.0 - (weighted_observed / weighted_expected)

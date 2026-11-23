"""
Common helpers for evaluation scripts.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


TRUE_VALUES = {"yes", "true", "1", "y"}
FALSE_VALUES = {"no", "false", "0", "n"}
PARTIAL_VALUES = {"partial", "partially", "partially_faithful", "inferred"}
EMPTY_VALUES = {"", "na", "n/a", "none", "null", "-"}


def norm(value: Any) -> str:
    """Normalize a CSV/JSON scalar for metric computation."""
    if value is None:
        return ""
    return str(value).strip().lower()


def is_filled(value: Any) -> bool:
    """Return True when a label cell contains a usable annotation."""
    return norm(value) not in EMPTY_VALUES


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def flatten_list_field(value: Any) -> str:
    """Render list-like fields for CSV output."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(x) for x in value)
    return str(value)


def parse_semicolon_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def group_by(rows: Iterable[dict[str, Any]], key_fn: Callable[[dict[str, Any]], Any]) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return dict(grouped)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return ";".join(str(x) for x in value)
    if isinstance(value, dict):
        import json
        return json.dumps(value, ensure_ascii=False)
    return value


def collect_csvs(paths: list[str | Path], patterns: list[str]) -> list[Path]:
    """
    Collect CSV files from explicit files or directories.
    """
    found: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix.lower() == ".csv":
            found.append(path)
        elif path.is_dir():
            for pattern in patterns:
                found.extend(path.rglob(pattern))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def rows_from_csvs(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_csv_rows(path))
    return rows
"""
Input/output helpers for reproducible experiment artifacts.

The pipeline writes JSONL files for machine processing and CSV files for
annotation. These helpers keep serialization behavior consistent across scripts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if needed and return it as a Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=indent, ensure_ascii=False)


def append_jsonl(record: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def write_jsonl(records: Iterable[Any], path: str | Path, append: bool = False) -> None:
    """
    Write records to JSONL.

    Parameters
    ----------
    records:
        Iterable of dictionaries or dataclass instances.
    path:
        Destination file.
    append:
        If True, append to the existing file. If False, overwrite it.
    """
    path = Path(path)
    ensure_dir(path.parent)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path, fieldnames: list[str] | None = None) -> None:
    rows = [_jsonable(row) for row in rows]
    path = Path(path)
    ensure_dir(path.parent)

    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    return value
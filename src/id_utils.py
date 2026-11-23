"""
Utilities for deterministic identifiers used by the pipeline.

The functions in this module intentionally avoid random UUIDs for artifacts that
must be compared across runs, such as documents, chunks, and attack steps.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def slugify(value: str, max_length: int = 80) -> str:
    """
    Convert a string into a filesystem- and JSON-friendly identifier fragment.

    Parameters
    ----------
    value:
        Input string.
    max_length:
        Maximum length of the returned slug.

    Returns
    -------
    str
        Lowercase slug containing only letters, digits, and underscores.
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = "item"
    return value[:max_length].strip("_") or "item"


def stable_hash(value: str, length: int = 10) -> str:
    """
    Return a deterministic short hash for the given value.
    """
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def make_document_id(file_path: str | Path, case_id: str | None = None) -> str:
    """
    Build a stable document identifier from a file path and optional case ID.
    """
    path = Path(file_path)
    stem = slugify(path.stem, max_length=60)
    digest = stable_hash(str(path.as_posix()), length=8)
    if case_id:
        return f"doc_{slugify(case_id, max_length=32)}_{stem}_{digest}"
    return f"doc_{stem}_{digest}"


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Build a stable chunk identifier from a document ID and chunk index.
    """
    return f"{document_id}_chunk{chunk_index:04d}"


def make_attack_step_id(case_id: str, tactic: str, step_index: int, name: str | None = None) -> str:
    """
    Build a deterministic attack-step identifier.
    """
    case_part = slugify(case_id, max_length=32)
    tactic_part = slugify(tactic, max_length=32)
    if name:
        name_part = slugify(name, max_length=36)
        return f"{case_part}_{tactic_part}_step{step_index:03d}_{name_part}"
    return f"{case_part}_{tactic_part}_step{step_index:03d}"


def make_condition_id(
    case_id: str,
    attack_step_id: str,
    condition_type: str,
    condition_index: int,
) -> str:
    """
    Build a deterministic condition identifier.
    """
    return (
        f"cond_{slugify(case_id, max_length=32)}_"
        f"{slugify(attack_step_id, max_length=80)}_"
        f"{slugify(condition_type, max_length=8)}_"
        f"{condition_index:03d}"
    )
"""
Local model inventory helpers.

For Ollama-backed local-first experiments, the raw output of `ollama list` is a
useful reproducibility artifact. The parser is best-effort because Ollama's
columns may vary across versions.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any


def collect_ollama_inventory(timeout_seconds: int = 10) -> dict[str, Any]:
    """Return raw and best-effort parsed `ollama list` information."""
    if shutil.which("ollama") is None:
        return {
            "ollama_available": False,
            "ollama_list_returncode": None,
            "ollama_list_stdout": "",
            "ollama_list_stderr": "ollama executable not found on PATH",
            "ollama_models": [],
        }

    try:
        completed = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception as exc:
        return {
            "ollama_available": True,
            "ollama_list_returncode": None,
            "ollama_list_stdout": "",
            "ollama_list_stderr": repr(exc),
            "ollama_models": [],
        }

    return {
        "ollama_available": True,
        "ollama_list_returncode": completed.returncode,
        "ollama_list_stdout": completed.stdout,
        "ollama_list_stderr": completed.stderr,
        "ollama_models": _parse_ollama_list(completed.stdout),
    }


def _parse_ollama_list(stdout: str) -> list[dict[str, str]]:
    lines = [line.rstrip() for line in stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []

    models: list[dict[str, str]] = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        # Typical columns: NAME ID SIZE MODIFIED. NAME is the only reliably
        # parseable field across versions; retain raw_line for reproducibility.
        models.append(
            {
                "name": parts[0],
                "raw_line": line,
            }
        )
    return models
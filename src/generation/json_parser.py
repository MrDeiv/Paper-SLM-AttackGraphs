"""
JSON parsing helpers for local model outputs.

Local SLMs sometimes wrap JSON in Markdown fences or include short preambles.
These helpers extract the first plausible JSON object while still failing loudly
when no usable JSON exists.
"""

from __future__ import annotations

import json
import re
from typing import Any


class JSONParseError(ValueError):
    """Raised when model output cannot be parsed as JSON."""


def parse_json_object(text: str) -> dict[str, Any]:
    """
    Parse a JSON object from model output.
    """
    text = (text or "").strip()
    if not text:
        raise JSONParseError("Cannot parse empty model output")

    candidates = _candidate_json_strings(text)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            last_error = exc

    raise JSONParseError(f"Could not parse JSON object. Last error: {last_error}. Output: {text[:500]}")


def _candidate_json_strings(text: str) -> list[str]:
    candidates: list[str] = []

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)

    candidates.append(text)

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first : last + 1])

    # Preserve order while removing exact duplicates.
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique
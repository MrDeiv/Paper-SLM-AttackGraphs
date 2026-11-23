"""
Deprecated verifier compatibility module.

The production pipeline described for the final implementation does not run a
condition verifier and does not emit support/directness/status labels. This file
is kept only so legacy imports do not break. Calling ConditionVerifier.verify()
returns the input condition unchanged.
"""

from __future__ import annotations

from typing import Any

from src.schemas import AttackStepRecord, GeneratedCondition, VerifiedCondition


class ConditionVerifier:
    """No-op compatibility wrapper."""

    def __init__(self, *, model_id: str = "none", temperature: float = 0.0, prompt_version: str = "disabled", num_ctx: int | None = None):
        self.model_id = model_id
        self.temperature = temperature
        self.prompt_version = prompt_version
        self.num_ctx = num_ctx

    def verify(
        self,
        *,
        attack_step: AttackStepRecord,
        condition: GeneratedCondition,
        retrieved_chunks: list[dict[str, Any]],
    ) -> VerifiedCondition:
        return condition


def pass_through_without_verification(condition: GeneratedCondition) -> VerifiedCondition:
    return condition

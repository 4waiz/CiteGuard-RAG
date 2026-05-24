"""Optional LLM-judge stub. **Disabled by default.**

This module intentionally does NOT call any paid API. It exposes a stable
interface so a future implementation can be slotted in. Enabling it requires
both a config flag and an explicit ``model`` name; otherwise we raise so the
pipeline never silently incurs an API call.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMJudgeResult:
    label: str        # SUPPORTED | UNSUPPORTED | CONTRADICTED | NOT_ENOUGH_EVIDENCE
    confidence: float
    rationale: str


class LLMJudge:
    def __init__(self, enabled: bool = False, model: str | None = None):
        self.enabled = bool(enabled)
        self.model = model

    def is_active(self) -> bool:
        return self.enabled and bool(self.model)

    def judge(self, claim_text: str, evidence_text: str) -> LLMJudgeResult:
        if not self.enabled:
            raise RuntimeError(
                "LLM judge is disabled. Enable it explicitly in config and set a model."
            )
        if not self.model:
            raise RuntimeError(
                "LLM judge is enabled but no model is configured. Set evaluators.llm_judge.model."
            )
        # Stub: an integrator must implement an API client here. We refuse to
        # invent a label because that would fabricate evaluator output.
        raise NotImplementedError(
            "LLMJudge.judge is a stub. Integrate an LLM client before using this module."
        )

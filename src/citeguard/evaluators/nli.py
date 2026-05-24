"""NLI evaluator.

Wraps a Hugging Face MNLI-compatible model. Supports both
``AutoModelForSequenceClassification`` (e.g. ``cross-encoder/nli-deberta-v3-base``)
and the standard MNLI checkpoints (``roberta-large-mnli``, etc.).

The label order is read from the model config and normalized to
{entailment, neutral, contradiction}.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class NLIResult:
    label: str            # "entailment" | "neutral" | "contradiction" | "n/a"
    score: float          # probability of the predicted label
    entail_prob: float
    neutral_prob: float
    contradict_prob: float


_DISABLED_RESULT = NLIResult(
    label="n/a", score=0.0, entail_prob=0.0, neutral_prob=0.0, contradict_prob=0.0
)


def _normalize_label(label: str) -> str:
    label = (label or "").lower()
    if "entail" in label:
        return "entailment"
    if "contradict" in label:
        return "contradiction"
    if "neutral" in label:
        return "neutral"
    return label


class NLIEvaluator:
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: str = "auto",
        max_evidence_chars: int = 1500,
        enabled: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.max_evidence_chars = max_evidence_chars
        self.enabled = enabled
        self._loaded = False
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._id2label: dict[int, str] = {}

    def _resolve_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        if self.device == "cuda":
            return "cuda"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        device = self._resolve_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(device)
        # Map model-specific label names back to the canonical NLI vocabulary.
        cfg_id2label = getattr(self._model.config, "id2label", {}) or {}
        self._id2label = {int(k): _normalize_label(str(v)) for k, v in cfg_id2label.items()}
        if not self._id2label:
            # Fallback to the conventional MNLI order.
            self._id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
        self._torch = torch
        self._loaded = True

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        if not self.enabled:
            return _DISABLED_RESULT
        if not premise or not hypothesis:
            return _DISABLED_RESULT
        self._ensure_loaded()
        premise = premise[: self.max_evidence_chars]

        torch = self._torch
        device = self._resolve_device()
        # cross-encoder NLI expects (premise, hypothesis).
        enc = self._tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        entail = neutral = contradict = 0.0
        for i, p in enumerate(probs):
            name = self._id2label.get(i, "")
            if name == "entailment":
                entail = float(p)
            elif name == "neutral":
                neutral = float(p)
            elif name == "contradiction":
                contradict = float(p)
        # Pick the largest of the three canonical classes.
        scored = {"entailment": entail, "neutral": neutral, "contradiction": contradict}
        label = max(scored.items(), key=lambda kv: kv[1])[0]
        return NLIResult(
            label=label,
            score=float(scored[label]),
            entail_prob=entail,
            neutral_prob=neutral,
            contradict_prob=contradict,
        )

    def predict_many(self, premises: list[str], hypothesis: str) -> list[NLIResult]:
        return [self.predict(p, hypothesis) for p in premises]

"""Semantic similarity evaluator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..retrieval.vector_store import VectorStore


@dataclass
class SemanticEvaluator:
    """Cosine similarity between a claim and a piece of evidence.

    Reuses an existing VectorStore so we don't reload the sentence-transformer
    per claim. Pass a freshly-built VectorStore if you only want the embedder.
    """

    vector_store: VectorStore

    def score(self, claim_text: str, evidence_text: str) -> float:
        if not claim_text or not evidence_text:
            return 0.0
        return float(self.vector_store.cosine(claim_text, evidence_text))

    def score_many(self, claim_text: str, evidence_texts: list[str]) -> list[float]:
        if not claim_text or not evidence_texts:
            return [0.0 for _ in evidence_texts]
        # Compute once for the claim, batch for evidence.
        q = self.vector_store.encode_query(claim_text)
        sbert = self.vector_store._sbert  # noqa: SLF001 — internal use
        mat = np.asarray(sbert.encode(evidence_texts, show_progress_bar=False)).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mat = mat / norms
        return [float(x) for x in mat @ q]

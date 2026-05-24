"""Per-example dense vector store.

Encodes the contexts once with a sentence-transformer and supports cosine
similarity queries. The transformer is loaded lazily and cached at module
level so repeated examples don't pay the load cost more than once.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..schemas import Context


_MODEL_CACHE: dict[tuple[str, str], "SentenceTransformerLike"] = {}


class SentenceTransformerLike:
    """Protocol-ish wrapper so we can swap in a fake model in tests."""

    def encode(self, sentences: Iterable[str], **kwargs) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def _resolve_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"
    # auto
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_sbert(model_name: str, device: str = "auto") -> "SentenceTransformerLike":
    """Return a (cached) sentence-transformer instance."""
    key = (model_name, _resolve_device(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    # Lazy import so unit tests that monkeypatch us don't pay the cost.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=key[1])
    _MODEL_CACHE[key] = model  # type: ignore[assignment]
    return model  # type: ignore[return-value]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


@dataclass
class DenseHit:
    doc_id: str
    score: float
    text: str
    title: str


class VectorStore:
    def __init__(
        self,
        contexts: list[Context],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        sbert: SentenceTransformerLike | None = None,
    ):
        self.contexts = contexts
        self._sbert = sbert if sbert is not None else get_sbert(model_name, device)
        if contexts:
            texts = [c.text or "" for c in contexts]
            vecs = np.asarray(self._sbert.encode(texts, show_progress_bar=False))
            self._matrix = _normalize(vecs.astype(np.float32))
        else:
            self._matrix = np.zeros((0, 1), dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        vec = np.asarray(self._sbert.encode([query or ""], show_progress_bar=False))
        return _normalize(vec.astype(np.float32))[0]

    def search(self, query: str, top_k: int = 5) -> list[DenseHit]:
        if not self.contexts or self._matrix.shape[0] == 0:
            return []
        q = self.encode_query(query)
        sims = (self._matrix @ q).astype(float)
        order = np.argsort(-sims)
        hits: list[DenseHit] = []
        for i in order[:top_k]:
            ctx = self.contexts[int(i)]
            hits.append(DenseHit(doc_id=ctx.doc_id, score=float(sims[int(i)]), text=ctx.text, title=ctx.title))
        return hits

    def cosine(self, query: str, doc_text: str) -> float:
        if not doc_text:
            return 0.0
        q = self.encode_query(query)
        d = np.asarray(self._sbert.encode([doc_text], show_progress_bar=False))
        d = _normalize(d.astype(np.float32))[0]
        return float(np.dot(q, d))

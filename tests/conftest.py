"""Shared test fixtures.

Provides a deterministic, dependency-free "fake" sentence-transformer so the
test suite doesn't require downloading a real model.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pytest


class FakeSBERT:
    """Hashes tokens into a fixed-dim bag-of-words vector.

    This is enough for the pipeline smoke test: the same words map to similar
    vectors, so claim-evidence cosine similarity is meaningful, while staying
    fully deterministic and zero-dependency.
    """

    dim = 64

    def encode(self, sentences: Iterable[str], **kwargs) -> np.ndarray:
        out = []
        for s in sentences:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in (s or "").lower().split():
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim
                v[h] += 1.0
            out.append(v)
        return np.asarray(out, dtype=np.float32)


@pytest.fixture
def fake_sbert() -> FakeSBERT:
    return FakeSBERT()

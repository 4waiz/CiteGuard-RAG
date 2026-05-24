"""Light-weight calibration utility.

We bucket claims by their support_score and report the empirical correctness
rate per bucket. This is a simple reliability check, not a full calibration
curve.
"""
from __future__ import annotations

import numpy as np


def support_score_buckets(
    support_scores: list[float],
    is_correct: list[bool],
    n_buckets: int = 10,
) -> list[dict]:
    if not support_scores:
        return []
    scores = np.asarray(support_scores, dtype=float)
    correct = np.asarray(is_correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    rows: list[dict] = []
    for i in range(n_buckets):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_buckets - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n > 0 else 0.0
        rows.append({"low": lo, "high": hi, "count": n, "accuracy": acc})
    return rows

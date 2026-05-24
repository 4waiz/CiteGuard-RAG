"""Retrieval metrics: recall@k vs gold support_doc_ids."""
from __future__ import annotations


def retrieval_recall_at_k(
    retrieved_by_claim: list[list[str]],
    gold_support_by_claim: list[list[str]],
    k: int = 5,
) -> float:
    """Macro recall@k.

    For each claim with a non-empty gold support set, recall@k is the
    fraction of gold doc ids present in the top-k retrieved ids. Claims with
    no gold support are skipped (no denominator).
    """
    assert len(retrieved_by_claim) == len(gold_support_by_claim)
    recalls: list[float] = []
    for retrieved, gold in zip(retrieved_by_claim, gold_support_by_claim):
        gold_set = set(gold)
        if not gold_set:
            continue
        top = retrieved[:k]
        hit = len(gold_set & set(top))
        recalls.append(hit / len(gold_set))
    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)

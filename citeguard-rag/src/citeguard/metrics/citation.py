"""Citation-level metrics.

We compare *cited* doc ids against the gold ``support_doc_ids`` per claim.

Precision = (cited ∩ gold) / cited
Recall    = (cited ∩ gold) / gold
We report mean over claims (macro), skipping claims where the denominator is
zero so we don't penalize an unciteable claim.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CitationMetrics:
    precision: float
    recall: float
    f1: float
    missing_citation_rate: float    # fraction of claims with no citation
    citation_mismatch_rate: float   # fraction of claims with citation but zero overlap with gold
    n_claims: int
    n_with_gold_support: int


def citation_metrics(
    cited_by_claim: list[list[str]],
    gold_support_by_claim: list[list[str]],
) -> CitationMetrics:
    assert len(cited_by_claim) == len(gold_support_by_claim)
    precisions: list[float] = []
    recalls: list[float] = []
    missing = 0
    mismatched = 0
    n_with_gold = 0
    for cited, gold in zip(cited_by_claim, gold_support_by_claim):
        cited_set = set(cited)
        gold_set = set(gold)
        if not cited_set:
            missing += 1
        if gold_set:
            n_with_gold += 1
            overlap = len(cited_set & gold_set)
            if cited_set:
                p = overlap / len(cited_set)
                precisions.append(p)
                if overlap == 0:
                    mismatched += 1
            r = overlap / len(gold_set)
            recalls.append(r)

    p = sum(precisions) / len(precisions) if precisions else 0.0
    r = sum(recalls) / len(recalls) if recalls else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    n = len(cited_by_claim)
    return CitationMetrics(
        precision=p,
        recall=r,
        f1=f1,
        missing_citation_rate=missing / n if n else 0.0,
        citation_mismatch_rate=mismatched / n if n else 0.0,
        n_claims=n,
        n_with_gold_support=n_with_gold,
    )

"""CSV table writers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schemas import ClaimEvaluation


CLAIM_EVAL_COLUMNS = [
    "example_id", "dataset", "question", "answer_id", "claim_id", "claim_text",
    "cited_doc_ids", "top_evidence_doc_ids", "best_evidence_doc_id", "best_evidence_text",
    "semantic_score", "bm25_score", "nli_label", "nli_score",
    "entail_prob", "contradict_prob", "neutral_prob",
    "support_score", "predicted_label", "gold_label", "is_correct",
    "latency_ms", "rationale",
]


def write_claim_eval_csv(evaluations: list[ClaimEvaluation], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not evaluations:
        pd.DataFrame(columns=CLAIM_EVAL_COLUMNS).to_csv(out, index=False)
        return out
    df = pd.DataFrame([e.to_row() for e in evaluations])
    for col in CLAIM_EVAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[CLAIM_EVAL_COLUMNS]
    df.to_csv(out, index=False)
    return out


EXAMPLE_SUMMARY_COLUMNS = [
    "example_id", "dataset", "num_claims", "supported_rate", "unsupported_rate",
    "contradiction_rate", "citation_mismatch_rate", "missing_citation_rate",
    "citation_precision", "citation_recall", "retrieval_recall_at_5", "avg_latency_ms",
]


def write_example_summary_csv(rows: list[dict], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame(columns=EXAMPLE_SUMMARY_COLUMNS).to_csv(out, index=False)
        return out
    df = pd.DataFrame(rows)
    for col in EXAMPLE_SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[EXAMPLE_SUMMARY_COLUMNS]
    df.to_csv(out, index=False)
    return out

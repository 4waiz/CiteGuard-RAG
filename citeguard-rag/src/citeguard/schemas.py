"""Typed schemas used across the pipeline.

We use dataclasses instead of Pydantic to keep the dependency surface small.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# Canonical label set. Code that compares labels should use these constants.
SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
CONTRADICTED = "CONTRADICTED"
CITATION_MISMATCH = "CITATION_MISMATCH"
SUPPORTED_UNCITED = "SUPPORTED_UNCITED"
NOT_ENOUGH_EVIDENCE = "NOT_ENOUGH_EVIDENCE"
PARSER_UNCERTAIN = "PARSER_UNCERTAIN"

LABELS = [
    SUPPORTED,
    UNSUPPORTED,
    CONTRADICTED,
    CITATION_MISMATCH,
    SUPPORTED_UNCITED,
    NOT_ENOUGH_EVIDENCE,
    PARSER_UNCERTAIN,
]


@dataclass
class Context:
    doc_id: str
    title: str
    text: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Context":
        return cls(
            doc_id=str(d.get("doc_id", "")),
            title=str(d.get("title", "")),
            text=str(d.get("text", "")),
        )


@dataclass
class GoldClaimLabel:
    claim_text: str
    label: str
    support_doc_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldClaimLabel":
        return cls(
            claim_text=str(d.get("claim_text", "")),
            label=str(d.get("label", "")),
            support_doc_ids=list(d.get("support_doc_ids", []) or []),
        )


@dataclass
class RAGExample:
    example_id: str
    dataset: str
    question: str
    answer: str
    contexts: list[Context]
    gold_claim_labels: list[GoldClaimLabel] = field(default_factory=list)
    is_synthetic: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RAGExample":
        contexts = [Context.from_dict(c) for c in d.get("contexts", []) or []]
        gold = [GoldClaimLabel.from_dict(g) for g in d.get("gold_claim_labels", []) or []]
        return cls(
            example_id=str(d.get("example_id", "")),
            dataset=str(d.get("dataset", "custom")),
            question=str(d.get("question", "")),
            answer=str(d.get("answer", "")),
            contexts=contexts,
            gold_claim_labels=gold,
            is_synthetic=bool(d.get("is_synthetic", False)),
        )


@dataclass
class Claim:
    claim_id: str          # e.g. "ex_001_c001"
    example_id: str
    text: str              # claim text with citation markers stripped
    raw_text: str          # original sentence including citation markers
    cited_doc_ids: list[str] = field(default_factory=list)
    parser_uncertain: bool = False


@dataclass
class ClaimEvaluation:
    claim_id: str
    example_id: str
    dataset: str
    question: str
    answer_id: str
    claim_text: str
    cited_doc_ids: list[str]
    top_evidence_doc_ids: list[str]
    best_evidence_doc_id: str
    best_evidence_text: str
    semantic_score: float        # cosine vs best evidence (-1..1)
    bm25_score: float            # raw BM25 score for the best lexical hit
    nli_label: str               # entailment | neutral | contradiction | n/a
    nli_score: float             # probability of the predicted nli label
    entail_prob: float
    contradict_prob: float
    neutral_prob: float
    support_score: float         # 0..1 hybrid support score
    predicted_label: str
    gold_label: str
    is_correct: bool
    latency_ms: float
    rationale: str

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        # CSV-friendly: serialize list fields as semicolon-joined strings.
        d["cited_doc_ids"] = ";".join(self.cited_doc_ids)
        d["top_evidence_doc_ids"] = ";".join(self.top_evidence_doc_ids)
        return d

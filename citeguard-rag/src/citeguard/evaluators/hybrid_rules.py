"""Hybrid diagnostic rules.

Given per-evidence semantic + NLI signals, decide one of:
SUPPORTED, UNSUPPORTED, CONTRADICTED, CITATION_MISMATCH,
SUPPORTED_UNCITED, NOT_ENOUGH_EVIDENCE, PARSER_UNCERTAIN.

The rules are intentionally explicit and traceable — each branch records a
rationale string so the dashboard can show *why* a label was assigned.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import ThresholdsConfig
from ..retrieval.evidence_mapper import EvidenceChunk, MappedEvidence
from ..schemas import (
    CITATION_MISMATCH,
    CONTRADICTED,
    NOT_ENOUGH_EVIDENCE,
    PARSER_UNCERTAIN,
    SUPPORTED,
    SUPPORTED_UNCITED,
    UNSUPPORTED,
)


@dataclass
class EvidenceSignal:
    chunk: EvidenceChunk
    semantic: float
    entail_prob: float
    contradict_prob: float
    neutral_prob: float
    nli_label: str   # "entailment" | "neutral" | "contradiction" | "n/a"
    nli_score: float


@dataclass
class DiagnosticResult:
    predicted_label: str
    rationale: str
    best_signal: EvidenceSignal | None
    support_score: float


def _support_score(sig: EvidenceSignal) -> float:
    """Combine semantic + entailment into a single 0..1 support score.

    Uses the average when both are present, else whichever is available.
    Semantic scores are clipped to [0,1] (cosine can be slightly negative).
    """
    parts: list[float] = []
    sem = max(0.0, min(1.0, sig.semantic))
    parts.append(sem)
    if sig.nli_label != "n/a":
        parts.append(max(0.0, min(1.0, sig.entail_prob)))
    return sum(parts) / len(parts) if parts else 0.0


def _best_supporting(signals: list[EvidenceSignal]) -> EvidenceSignal | None:
    if not signals:
        return None
    return max(signals, key=_support_score)


def _best_contradicting(signals: list[EvidenceSignal]) -> EvidenceSignal | None:
    if not signals:
        return None
    return max(signals, key=lambda s: s.contradict_prob)


class HybridDiagnoser:
    def __init__(self, thresholds: ThresholdsConfig):
        self.t = thresholds

    def diagnose(
        self,
        mapping: MappedEvidence,
        cited_signals: list[EvidenceSignal],
        retrieved_signals: list[EvidenceSignal],
    ) -> DiagnosticResult:
        claim = mapping.claim
        t = self.t

        # 1) Parser uncertain trumps everything (e.g. citation parsing failed on this claim).
        if claim.parser_uncertain:
            return DiagnosticResult(
                predicted_label=PARSER_UNCERTAIN,
                rationale="Citation parser could not extract a usable doc id from the claim.",
                best_signal=None,
                support_score=0.0,
            )

        all_signals = cited_signals + retrieved_signals

        # 2) Strong contradiction in cited evidence -> CONTRADICTED.
        worst_cited_contra = _best_contradicting(cited_signals)
        if worst_cited_contra is not None and worst_cited_contra.contradict_prob >= t.contradict:
            return DiagnosticResult(
                predicted_label=CONTRADICTED,
                rationale=(
                    f"Cited evidence ({worst_cited_contra.chunk.doc_id}) contradicts the claim "
                    f"(contradiction prob={worst_cited_contra.contradict_prob:.2f})."
                ),
                best_signal=worst_cited_contra,
                support_score=_support_score(worst_cited_contra),
            )
        # Contradiction in any retrieved evidence also counts.
        worst_any_contra = _best_contradicting(all_signals)
        if worst_any_contra is not None and worst_any_contra.contradict_prob >= t.contradict:
            # Only escalate to CONTRADICTED if no cited evidence supports it.
            best_cited = _best_supporting(cited_signals)
            cited_supports = (
                best_cited is not None
                and best_cited.entail_prob >= t.entail_support
                and best_cited.semantic >= t.semantic_related
            )
            if not cited_supports:
                return DiagnosticResult(
                    predicted_label=CONTRADICTED,
                    rationale=(
                        f"Retrieved evidence ({worst_any_contra.chunk.doc_id}) contradicts the claim "
                        f"(contradiction prob={worst_any_contra.contradict_prob:.2f})."
                    ),
                    best_signal=worst_any_contra,
                    support_score=_support_score(worst_any_contra),
                )

        # 3) Cited evidence supports claim -> SUPPORTED.
        best_cited = _best_supporting(cited_signals)
        cited_supports = bool(
            best_cited
            and (
                (best_cited.nli_label == "entailment" and best_cited.entail_prob >= t.entail_support)
                or (best_cited.nli_label == "n/a" and best_cited.semantic >= t.semantic_strong)
            )
            and best_cited.semantic >= t.semantic_related
        )
        if cited_supports:
            assert best_cited is not None
            return DiagnosticResult(
                predicted_label=SUPPORTED,
                rationale=(
                    f"Cited evidence ({best_cited.chunk.doc_id}) entails the claim "
                    f"(entail={best_cited.entail_prob:.2f}, cosine={best_cited.semantic:.2f})."
                ),
                best_signal=best_cited,
                support_score=_support_score(best_cited),
            )

        # 4) No citation: see whether retrieved evidence supports the claim.
        best_retrieved = _best_supporting(retrieved_signals)
        retrieved_supports = bool(
            best_retrieved
            and (
                (best_retrieved.nli_label == "entailment" and best_retrieved.entail_prob >= t.entail_support)
                or (best_retrieved.nli_label == "n/a" and best_retrieved.semantic >= t.semantic_strong)
            )
            and best_retrieved.semantic >= t.semantic_related
        )

        if not claim.cited_doc_ids:
            if retrieved_supports:
                assert best_retrieved is not None
                return DiagnosticResult(
                    predicted_label=SUPPORTED_UNCITED,
                    rationale=(
                        f"No citation, but retrieved evidence ({best_retrieved.chunk.doc_id}) "
                        f"supports the claim (entail={best_retrieved.entail_prob:.2f}, "
                        f"cosine={best_retrieved.semantic:.2f})."
                    ),
                    best_signal=best_retrieved,
                    support_score=_support_score(best_retrieved),
                )
            # Truly nothing in the corpus supports it.
            if best_retrieved is None or best_retrieved.semantic < t.unsupported_floor:
                return DiagnosticResult(
                    predicted_label=UNSUPPORTED,
                    rationale="No citation provided and no retrieved evidence is even semantically related.",
                    best_signal=best_retrieved,
                    support_score=_support_score(best_retrieved) if best_retrieved else 0.0,
                )
            return DiagnosticResult(
                predicted_label=NOT_ENOUGH_EVIDENCE,
                rationale=(
                    "No citation provided; retrieved evidence is related but does not clearly support the claim."
                ),
                best_signal=best_retrieved,
                support_score=_support_score(best_retrieved),
            )

        # 5) Citation exists but cited evidence does not support, retrieved does -> CITATION_MISMATCH.
        if retrieved_supports:
            assert best_retrieved is not None
            return DiagnosticResult(
                predicted_label=CITATION_MISMATCH,
                rationale=(
                    f"Cited docs do not support the claim, but retrieved doc "
                    f"({best_retrieved.chunk.doc_id}) does "
                    f"(entail={best_retrieved.entail_prob:.2f}, cosine={best_retrieved.semantic:.2f})."
                ),
                best_signal=best_retrieved,
                support_score=_support_score(best_retrieved),
            )

        # 6) Otherwise: NOT_ENOUGH_EVIDENCE vs UNSUPPORTED based on semantic floor.
        candidate = best_cited or best_retrieved
        if candidate is None or _support_score(candidate) < t.unsupported_floor:
            return DiagnosticResult(
                predicted_label=UNSUPPORTED,
                rationale="No evidence (cited or retrieved) supports the claim.",
                best_signal=candidate,
                support_score=_support_score(candidate) if candidate else 0.0,
            )
        return DiagnosticResult(
            predicted_label=NOT_ENOUGH_EVIDENCE,
            rationale=(
                "Evidence is related but neither cited nor retrieved chunks clearly entail the claim."
            ),
            best_signal=candidate,
            support_score=_support_score(candidate),
        )

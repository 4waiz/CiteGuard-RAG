"""End-to-end CiteGuard-RAG pipeline.

Given a list of RAGExamples and a config, produce:
- a list of ClaimEvaluation rows (one per claim)
- a list of per-example summary dicts
- a list of evidence-card dicts (one per claim) for the dashboard
- aggregate metrics
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import CiteGuardConfig
from .evaluators.hybrid_rules import (
    DiagnosticResult,
    EvidenceSignal,
    HybridDiagnoser,
)
from .evaluators.nli import NLIEvaluator, NLIResult
from .evaluators.semantic import SemanticEvaluator
from .metrics.citation import citation_metrics
from .metrics.classification import classification_report_dict, confusion_matrix
from .metrics.retrieval import retrieval_recall_at_k
from .preprocess.claim_splitter import ClaimSplitter
from .preprocess.citation_parser import CitationParser
from .retrieval.evidence_mapper import EvidenceMapper, MappedEvidence
from .retrieval.vector_store import VectorStore
from .schemas import (
    CITATION_MISMATCH,
    CONTRADICTED,
    Claim,
    ClaimEvaluation,
    PARSER_UNCERTAIN,
    RAGExample,
    SUPPORTED,
    SUPPORTED_UNCITED,
    UNSUPPORTED,
)


def _gold_for_claim(example: RAGExample, claim: Claim) -> tuple[str, list[str]]:
    """Match a predicted claim to its gold label by text similarity.

    We use a simple lowercase substring or prefix overlap heuristic — gold
    text rarely matches the splitter's output exactly. Returns ("", [])
    when no gold label is provided.
    """
    if not example.gold_claim_labels:
        return "", []
    pred = (claim.text or "").lower().strip()
    if not pred:
        return "", []
    best_label = ""
    best_support: list[str] = []
    best_score = 0.0
    for g in example.gold_claim_labels:
        gold = (g.claim_text or "").lower().strip()
        if not gold:
            continue
        # Jaccard over whitespace tokens.
        a, b = set(pred.split()), set(gold.split())
        if not a or not b:
            continue
        inter = len(a & b)
        union = len(a | b)
        jacc = inter / union if union else 0.0
        if jacc > best_score:
            best_score = jacc
            best_label = g.label
            best_support = list(g.support_doc_ids)
    if best_score < 0.25:
        return "", []
    return best_label, best_support


class CiteGuardPipeline:
    def __init__(self, config: CiteGuardConfig):
        self.config = config
        self._splitter = ClaimSplitter(
            min_chars=config.claim_splitter.min_chars,
            strip_citations_for_embedding=config.claim_splitter.strip_citations_for_embedding,
            citation_parser=CitationParser(
                patterns=config.citation_parser.patterns,
                numeric_prefix=config.citation_parser.numeric_prefix,
            ),
        )
        self._diagnoser = HybridDiagnoser(config.thresholds)
        self._nli: NLIEvaluator | None = None
        # We lazily create NLI only if any evaluator config wants it.
        if config.evaluators.nli.enabled:
            self._nli = NLIEvaluator(
                model_name=config.models.nli,
                device=config.models.device,
                max_evidence_chars=config.evaluators.nli.max_evidence_chars,
                enabled=True,
            )

    # ------------------------------------------------------------------ public
    def run(
        self,
        examples: list[RAGExample],
        show_progress: bool = True,
    ) -> dict[str, Any]:
        evaluations: list[ClaimEvaluation] = []
        example_summaries: list[dict] = []
        evidence_cards: list[dict] = []

        iterator = tqdm(examples, disable=not show_progress, desc="examples")
        for example in iterator:
            ex_evals, ex_summary, ex_cards = self._process_example(example)
            evaluations.extend(ex_evals)
            example_summaries.append(ex_summary)
            evidence_cards.extend(ex_cards)

        aggregate = self._aggregate(evaluations)
        return {
            "evaluations": evaluations,
            "example_summaries": example_summaries,
            "evidence_cards": evidence_cards,
            "aggregate": aggregate,
        }

    def write_outputs(self, results: dict[str, Any]) -> dict[str, Path]:
        from .reporting.tables import write_claim_eval_csv, write_example_summary_csv

        out = self.config.output
        tables_dir = Path(out.tables_dir)
        cards_dir = Path(out.cards_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        cards_dir.mkdir(parents=True, exist_ok=True)

        claim_eval_path = tables_dir / "claim_eval.csv"
        write_claim_eval_csv(results["evaluations"], claim_eval_path)

        summary_path = tables_dir / "example_summary.csv"
        write_example_summary_csv(results["example_summaries"], summary_path)

        cards_path = cards_dir / "evidence_cards.jsonl"
        with open(cards_path, "w", encoding="utf-8") as f:
            for card in results["evidence_cards"]:
                f.write(json.dumps(card, ensure_ascii=False) + "\n")

        agg_path = tables_dir / "aggregate_metrics.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(results["aggregate"], f, indent=2, default=str)

        return {
            "claim_eval": claim_eval_path,
            "example_summary": summary_path,
            "evidence_cards": cards_path,
            "aggregate_metrics": agg_path,
        }

    # --------------------------------------------------------------- internals
    def _process_example(
        self, example: RAGExample
    ) -> tuple[list[ClaimEvaluation], dict, list[dict]]:
        claims = self._splitter.split(example.example_id, example.answer)
        ex_start = time.perf_counter()

        # Build retrieval indices once per example.
        use_dense = self.config.retrieval.use_dense
        mapper = EvidenceMapper(
            contexts=example.contexts,
            top_k=self.config.retrieval.top_k,
            use_bm25=self.config.retrieval.use_bm25,
            use_dense=use_dense,
            sbert_model=self.config.models.sbert,
            device=self.config.models.device,
        )
        # Reuse the mapper's VectorStore for semantic scoring if dense is on,
        # else stand up a private one just for the semantic evaluator (needed
        # so we can compute cosines even when retrieval skips dense).
        sem_eval: SemanticEvaluator | None = None
        if self.config.evaluators.semantic.enabled:
            vs = mapper.vector_store
            if vs is None:
                vs = VectorStore(
                    example.contexts,
                    model_name=self.config.models.sbert,
                    device=self.config.models.device,
                )
            sem_eval = SemanticEvaluator(vector_store=vs)

        ex_evals: list[ClaimEvaluation] = []
        ex_cards: list[dict] = []
        latencies: list[float] = []
        cited_per_claim: list[list[str]] = []
        gold_support_per_claim: list[list[str]] = []
        retrieved_per_claim: list[list[str]] = []

        for claim in claims:
            t0 = time.perf_counter()
            mapping = mapper.map(claim)
            cited_signals = self._signals_for(mapping, mapping.cited_chunks, sem_eval, claim)
            retrieved_signals = self._signals_for(
                mapping, mapping.retrieved_chunks, sem_eval, claim
            )
            diag: DiagnosticResult = self._diagnoser.diagnose(
                mapping, cited_signals, retrieved_signals
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            gold_label, gold_support = _gold_for_claim(example, claim)

            best_signal = diag.best_signal
            best_doc_id = best_signal.chunk.doc_id if best_signal else ""
            best_text = best_signal.chunk.text if best_signal else ""
            nli_label = best_signal.nli_label if best_signal else "n/a"
            nli_score = best_signal.nli_score if best_signal else 0.0
            entail = best_signal.entail_prob if best_signal else 0.0
            contra = best_signal.contradict_prob if best_signal else 0.0
            neutral = best_signal.neutral_prob if best_signal else 0.0
            semantic = best_signal.semantic if best_signal else 0.0
            bm25 = best_signal.chunk.bm25_score if best_signal else 0.0

            top_retrieved_ids = mapping.top_evidence_doc_ids()

            evaluation = ClaimEvaluation(
                claim_id=claim.claim_id,
                example_id=example.example_id,
                dataset=example.dataset,
                question=example.question,
                answer_id=example.example_id,
                claim_text=claim.text,
                cited_doc_ids=claim.cited_doc_ids,
                top_evidence_doc_ids=top_retrieved_ids,
                best_evidence_doc_id=best_doc_id,
                best_evidence_text=best_text,
                semantic_score=float(semantic),
                bm25_score=float(bm25),
                nli_label=nli_label,
                nli_score=float(nli_score),
                entail_prob=float(entail),
                contradict_prob=float(contra),
                neutral_prob=float(neutral),
                support_score=float(diag.support_score),
                predicted_label=diag.predicted_label,
                gold_label=gold_label,
                is_correct=bool(gold_label) and gold_label == diag.predicted_label,
                latency_ms=float(latency_ms),
                rationale=diag.rationale,
            )
            ex_evals.append(evaluation)
            ex_cards.append(self._build_card(example, claim, mapping, cited_signals, retrieved_signals, evaluation))
            cited_per_claim.append(claim.cited_doc_ids)
            gold_support_per_claim.append(gold_support)
            retrieved_per_claim.append(top_retrieved_ids)

        ex_total_ms = (time.perf_counter() - ex_start) * 1000.0

        summary = self._summarize_example(
            example, ex_evals, cited_per_claim, gold_support_per_claim, retrieved_per_claim, ex_total_ms
        )
        return ex_evals, summary, ex_cards

    def _signals_for(
        self,
        mapping: MappedEvidence,
        chunks,
        sem_eval: SemanticEvaluator | None,
        claim: Claim,
    ) -> list[EvidenceSignal]:
        signals: list[EvidenceSignal] = []
        for chunk in chunks:
            sem = 0.0
            if sem_eval is not None and chunk.text:
                sem = sem_eval.score(claim.text, chunk.text)
            if self._nli is not None and chunk.text:
                nli: NLIResult = self._nli.predict(chunk.text, claim.text)
            else:
                nli = NLIResult(label="n/a", score=0.0, entail_prob=0.0, neutral_prob=0.0, contradict_prob=0.0)
            signals.append(
                EvidenceSignal(
                    chunk=chunk,
                    semantic=sem,
                    entail_prob=nli.entail_prob,
                    contradict_prob=nli.contradict_prob,
                    neutral_prob=nli.neutral_prob,
                    nli_label=nli.label,
                    nli_score=nli.score,
                )
            )
        return signals

    def _build_card(
        self,
        example: RAGExample,
        claim: Claim,
        mapping: MappedEvidence,
        cited_signals: list[EvidenceSignal],
        retrieved_signals: list[EvidenceSignal],
        evaluation: ClaimEvaluation,
    ) -> dict:
        def signal_view(sig: EvidenceSignal, source: str) -> dict:
            return {
                "doc_id": sig.chunk.doc_id,
                "title": sig.chunk.title,
                "text": sig.chunk.text,
                "source": source,
                "semantic_score": float(sig.semantic),
                "bm25_score": float(sig.chunk.bm25_score),
                "dense_score": float(sig.chunk.dense_score),
                "nli_label": sig.nli_label,
                "entail_prob": float(sig.entail_prob),
                "neutral_prob": float(sig.neutral_prob),
                "contradict_prob": float(sig.contradict_prob),
            }

        return {
            "claim_id": claim.claim_id,
            "example_id": example.example_id,
            "dataset": example.dataset,
            "question": example.question,
            "answer": example.answer,
            "claim_text": claim.text,
            "raw_claim_text": claim.raw_text,
            "cited_doc_ids": claim.cited_doc_ids,
            "predicted_label": evaluation.predicted_label,
            "gold_label": evaluation.gold_label,
            "support_score": evaluation.support_score,
            "rationale": evaluation.rationale,
            "latency_ms": evaluation.latency_ms,
            "best_evidence": {
                "doc_id": evaluation.best_evidence_doc_id,
                "text": evaluation.best_evidence_text,
            },
            "cited_evidence": [signal_view(s, "cited") for s in cited_signals],
            "retrieved_evidence": [signal_view(s, "retrieved") for s in retrieved_signals],
            "is_synthetic": example.is_synthetic,
        }

    def _summarize_example(
        self,
        example: RAGExample,
        ex_evals: list[ClaimEvaluation],
        cited_per_claim: list[list[str]],
        gold_support_per_claim: list[list[str]],
        retrieved_per_claim: list[list[str]],
        total_ms: float,
    ) -> dict:
        n = len(ex_evals)

        def rate(label: str) -> float:
            if not n:
                return 0.0
            return sum(1 for e in ex_evals if e.predicted_label == label) / n

        cit = citation_metrics(cited_per_claim, gold_support_per_claim)
        recall5 = retrieval_recall_at_k(retrieved_per_claim, gold_support_per_claim, k=5)
        avg_lat = (sum(e.latency_ms for e in ex_evals) / n) if n else 0.0

        return {
            "example_id": example.example_id,
            "dataset": example.dataset,
            "num_claims": n,
            "supported_rate": rate(SUPPORTED),
            "unsupported_rate": rate(UNSUPPORTED),
            "contradiction_rate": rate(CONTRADICTED),
            "citation_mismatch_rate": rate(CITATION_MISMATCH),
            "missing_citation_rate": cit.missing_citation_rate,
            "citation_precision": cit.precision,
            "citation_recall": cit.recall,
            "retrieval_recall_at_5": recall5,
            "avg_latency_ms": avg_lat,
            "total_latency_ms": total_ms,
            "supported_uncited_rate": rate(SUPPORTED_UNCITED),
            "parser_uncertain_rate": rate(PARSER_UNCERTAIN),
        }

    def _aggregate(self, evaluations: list[ClaimEvaluation]) -> dict:
        gold_pairs = [(e.gold_label, e.predicted_label) for e in evaluations if e.gold_label]
        if gold_pairs:
            y_true = [p[0] for p in gold_pairs]
            y_pred = [p[1] for p in gold_pairs]
            cls = classification_report_dict(y_true, y_pred)
            cm, labels = confusion_matrix(y_true, y_pred)
            confusion = {"labels": labels, "matrix": cm.tolist()}
        else:
            cls = classification_report_dict([], [])
            confusion = {"labels": [], "matrix": []}

        cited = [list(e.cited_doc_ids) for e in evaluations]
        gold_support = [[] for _ in evaluations]
        retrieved = [list(e.top_evidence_doc_ids) for e in evaluations]
        cit = citation_metrics(cited, gold_support)

        return {
            "n_claims": len(evaluations),
            "n_with_gold_label": len(gold_pairs),
            "macro_f1": cls["macro_f1"],
            "weighted_f1": cls["weighted_f1"],
            "per_label": cls["per_label"],
            "sklearn_report": cls["sklearn_report"],
            "confusion": confusion,
            "missing_citation_rate": cit.missing_citation_rate,
            "citation_mismatch_rate": cit.citation_mismatch_rate,
            "avg_latency_ms": (
                sum(e.latency_ms for e in evaluations) / len(evaluations)
                if evaluations
                else 0.0
            ),
        }

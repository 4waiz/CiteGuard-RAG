"""Evidence mapper: turns a claim into cited+retrieved evidence chunks."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import Claim, Context
from .bm25 import BM25Retriever, BM25Hit
from .vector_store import VectorStore, DenseHit


@dataclass
class EvidenceChunk:
    doc_id: str
    text: str
    title: str
    bm25_score: float = 0.0
    dense_score: float = 0.0
    source: str = "retrieved"  # "cited" or "retrieved"


@dataclass
class MappedEvidence:
    claim: Claim
    cited_chunks: list[EvidenceChunk] = field(default_factory=list)
    retrieved_chunks: list[EvidenceChunk] = field(default_factory=list)

    def all_chunks(self) -> list[EvidenceChunk]:
        seen = set()
        out: list[EvidenceChunk] = []
        for c in self.cited_chunks + self.retrieved_chunks:
            key = (c.doc_id, c.text)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def top_evidence_doc_ids(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for c in self.retrieved_chunks:
            if c.doc_id not in seen:
                seen.add(c.doc_id)
                out.append(c.doc_id)
        return out


def _min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class EvidenceMapper:
    """Joins cited evidence with BM25 + dense retrieval for a single example."""

    def __init__(
        self,
        contexts: list[Context],
        top_k: int = 5,
        use_bm25: bool = True,
        use_dense: bool = True,
        sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        vector_store: VectorStore | None = None,
    ):
        self.contexts = contexts
        self.top_k = top_k
        self.use_bm25 = use_bm25
        self.use_dense = use_dense
        self._by_doc = {c.doc_id: c for c in contexts}
        self._bm25 = BM25Retriever(contexts) if use_bm25 else None
        if use_dense:
            self._vec = vector_store or VectorStore(contexts, model_name=sbert_model, device=device)
        else:
            self._vec = None

    @property
    def vector_store(self) -> VectorStore | None:
        return self._vec

    def map(self, claim: Claim) -> MappedEvidence:
        cited_chunks: list[EvidenceChunk] = []
        for doc_id in claim.cited_doc_ids:
            ctx = self._by_doc.get(doc_id)
            if ctx is None:
                # Citation points at a doc the example never provided. Keep
                # a placeholder so downstream rules can detect the mismatch.
                cited_chunks.append(EvidenceChunk(doc_id=doc_id, text="", title="", source="cited"))
            else:
                cited_chunks.append(
                    EvidenceChunk(doc_id=ctx.doc_id, text=ctx.text, title=ctx.title, source="cited")
                )

        bm25_hits: list[BM25Hit] = []
        if self._bm25 is not None:
            bm25_hits = self._bm25.search(claim.text, top_k=self.top_k)

        dense_hits: list[DenseHit] = []
        if self._vec is not None:
            dense_hits = self._vec.search(claim.text, top_k=self.top_k)

        # Fuse retrieved hits: min-max normalize each then average.
        retrieved = self._fuse(bm25_hits, dense_hits)
        return MappedEvidence(claim=claim, cited_chunks=cited_chunks, retrieved_chunks=retrieved)

    def _fuse(self, bm25_hits: list[BM25Hit], dense_hits: list[DenseHit]) -> list[EvidenceChunk]:
        # Combine on doc_id. Each retriever contributes a normalized score.
        bm_scores = {h.doc_id: h.score for h in bm25_hits}
        dn_scores = {h.doc_id: h.score for h in dense_hits}
        bm_norm_list = _min_max(list(bm_scores.values()))
        dn_norm_list = _min_max(list(dn_scores.values()))
        bm_norm = dict(zip(bm_scores.keys(), bm_norm_list))
        dn_norm = dict(zip(dn_scores.keys(), dn_norm_list))

        ids = list({*bm_scores.keys(), *dn_scores.keys()})
        scored: list[tuple[str, float]] = []
        for doc_id in ids:
            parts = []
            if doc_id in bm_norm:
                parts.append(bm_norm[doc_id])
            if doc_id in dn_norm:
                parts.append(dn_norm[doc_id])
            fused = sum(parts) / max(len(parts), 1)
            scored.append((doc_id, fused))
        scored.sort(key=lambda x: -x[1])

        chunks: list[EvidenceChunk] = []
        for doc_id, _fused in scored[: self.top_k]:
            ctx = self._by_doc.get(doc_id)
            if ctx is None:
                continue
            chunks.append(
                EvidenceChunk(
                    doc_id=doc_id,
                    text=ctx.text,
                    title=ctx.title,
                    bm25_score=float(bm_scores.get(doc_id, 0.0)),
                    dense_score=float(dn_scores.get(doc_id, 0.0)),
                    source="retrieved",
                )
            )
        return chunks

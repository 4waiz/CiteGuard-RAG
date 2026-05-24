"""Per-example BM25 retriever.

The context set is small (a handful of docs per example) so we build a fresh
BM25 index per example rather than maintaining a global one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from ..schemas import Context


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class BM25Hit:
    doc_id: str
    score: float
    text: str
    title: str


class BM25Retriever:
    """Lexical retriever over a list of Context objects."""

    def __init__(self, contexts: list[Context]):
        self.contexts = contexts
        self._tokenized = [_tokenize(c.text) for c in contexts]
        # rank-bm25 errors on an empty corpus; guard the edge case.
        if self._tokenized and any(self._tokenized):
            self._bm25: BM25Okapi | None = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = 5) -> list[BM25Hit]:
        if not self.contexts or self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
        out: list[BM25Hit] = []
        for i in order[:top_k]:
            ctx = self.contexts[i]
            out.append(BM25Hit(doc_id=ctx.doc_id, score=float(scores[i]), text=ctx.text, title=ctx.title))
        return out

"""Rule-based sentence/claim splitter.

We deliberately keep this simple: regex sentence splitting plus citation
parsing. The hard cases (coordinated clauses, citations spanning sentences)
can be handled later if the prototype shows it's worth it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import Claim
from .citation_parser import CitationParser
from .text_cleaner import clean_text, strip_citation_markers


# A naive sentence boundary detector: split on .!? followed by whitespace.
# We use a lookbehind/lookahead so the terminator stays attached.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\(\[])")


def _split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


@dataclass
class ClaimSplitter:
    min_chars: int = 5
    strip_citations_for_embedding: bool = True
    citation_parser: CitationParser | None = None

    def __post_init__(self) -> None:
        if self.citation_parser is None:
            self.citation_parser = CitationParser()

    def split(self, example_id: str, answer: str) -> list[Claim]:
        sentences = _split_sentences(answer)
        claims: list[Claim] = []
        idx = 0
        for raw in sentences:
            if len(raw) < self.min_chars:
                continue
            idx += 1
            cited_ids, parser_uncertain = self.citation_parser.parse(raw)
            text_for_embed = (
                strip_citation_markers(raw)
                if self.strip_citations_for_embedding
                else clean_text(raw)
            )
            # If stripping citations emptied the claim (e.g. "[1].") mark uncertain.
            if not text_for_embed:
                parser_uncertain = True
                text_for_embed = clean_text(raw)
            claims.append(
                Claim(
                    claim_id=f"{example_id}_c{idx:03d}",
                    example_id=example_id,
                    text=text_for_embed,
                    raw_text=raw,
                    cited_doc_ids=cited_ids,
                    parser_uncertain=parser_uncertain,
                )
            )
        return claims


def split_into_claims(
    example_id: str,
    answer: str,
    min_chars: int = 5,
    strip_citations_for_embedding: bool = True,
) -> list[Claim]:
    splitter = ClaimSplitter(
        min_chars=min_chars,
        strip_citations_for_embedding=strip_citations_for_embedding,
    )
    return splitter.split(example_id, answer)

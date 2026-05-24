"""Citation parsing.

Supports bracketed citations such as ``[doc1]``, ``[1]``, ``[doc1, doc2]``.
Numeric citations like ``[1]`` are normalized to ``doc1`` so they line up
with context ``doc_id``s when authors number their sources.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_DEFAULT_PATTERNS = [r"\[([^\]\[]*)\]"]
_NUMERIC_RE = re.compile(r"^\d+$")


@dataclass
class CitationParser:
    """Parse cited doc ids from a sentence.

    ``numeric_prefix`` controls how ``[1]`` is rendered. With the default
    ``"doc"`` it becomes ``"doc1"``.
    """

    patterns: list[str] | None = None
    numeric_prefix: str = "doc"

    def __post_init__(self) -> None:
        patterns = self.patterns or _DEFAULT_PATTERNS
        self._compiled = [re.compile(p) for p in patterns]

    def parse(self, sentence: str) -> tuple[list[str], bool]:
        """Return (cited_doc_ids, parser_uncertain).

        ``parser_uncertain`` is True when bracketed content was found but
        produced no usable doc ids (e.g. ``[?]``, ``[ ]``, ``[see fig 1]``).
        Empty input or input with no brackets is not uncertain — just uncited.
        """
        if not sentence:
            return [], False

        found_any_bracket = False
        produced_ids: list[str] = []
        seen: set[str] = set()

        for regex in self._compiled:
            for match in regex.finditer(sentence):
                found_any_bracket = True
                inner = match.group(1).strip()
                if not inner:
                    continue
                # Split on commas or semicolons inside the bracket.
                parts = re.split(r"[,;]", inner)
                for raw in parts:
                    tok = raw.strip()
                    if not tok:
                        continue
                    doc_id = self._normalize(tok)
                    if doc_id and doc_id not in seen:
                        produced_ids.append(doc_id)
                        seen.add(doc_id)

        parser_uncertain = found_any_bracket and not produced_ids
        return produced_ids, parser_uncertain

    def _normalize(self, token: str) -> str | None:
        token = token.strip()
        if not token:
            return None
        if _NUMERIC_RE.match(token):
            return f"{self.numeric_prefix}{token}"
        # Drop trailing punctuation like "doc1." but keep underscores/hyphens.
        token = re.sub(r"[^\w\-]+$", "", token)
        token = re.sub(r"^[^\w]+", "", token)
        if not token:
            return None
        # Reject obviously-non-citation content like "see Fig 2" (contains spaces).
        if " " in token:
            # Allow "Smith2020" but reject "see fig" — i.e. require no whitespace.
            return None
        return token


def parse_citations_in_sentence(
    sentence: str,
    patterns: list[str] | None = None,
    numeric_prefix: str = "doc",
) -> tuple[list[str], bool]:
    """Convenience wrapper that constructs a parser per call."""
    return CitationParser(patterns=patterns, numeric_prefix=numeric_prefix).parse(sentence)

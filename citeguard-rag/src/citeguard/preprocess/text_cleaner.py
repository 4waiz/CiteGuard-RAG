"""Light text cleaning utilities."""
from __future__ import annotations

import re

# Matches bracket citations like [1], [doc2], [doc1, doc2], [Smith 2020].
CITATION_RE = re.compile(r"\[[^\]\[]+\]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Collapse whitespace and strip surrounding spaces."""
    if not text:
        return ""
    return WHITESPACE_RE.sub(" ", text).strip()


def strip_citation_markers(text: str) -> str:
    """Remove bracketed citation markers from the text and tidy whitespace.

    Used before embedding a claim so citation tokens do not bias similarity.
    """
    if not text:
        return ""
    stripped = CITATION_RE.sub("", text)
    return clean_text(stripped)

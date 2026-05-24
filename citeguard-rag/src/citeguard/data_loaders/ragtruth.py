"""RAGTruth loader (skeleton).

RAGTruth is not redistributed with this repository. To use it:
1. Download from the official source (the dataset's authors).
2. Convert each record to the universal JSONL schema described in the README.
3. Point ``load_ragtruth`` at the converted file.

If the converted file does not exist this function raises FileNotFoundError
rather than fabricating examples.
"""
from __future__ import annotations

from pathlib import Path

from ..schemas import RAGExample
from .custom_jsonl import load_jsonl


def load_ragtruth(path: str | Path) -> list[RAGExample]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"RAGTruth file not found at {p}. Convert RAGTruth to the universal "
            "JSONL schema (see README) and place it at this path."
        )
    examples = load_jsonl(p)
    for ex in examples:
        if not ex.dataset:
            ex.dataset = "ragtruth"
    return examples

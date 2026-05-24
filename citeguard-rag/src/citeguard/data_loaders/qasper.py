"""QASPER loader (skeleton). See README for the conversion path."""
from __future__ import annotations

from pathlib import Path

from ..schemas import RAGExample
from .custom_jsonl import load_jsonl


def load_qasper(path: str | Path) -> list[RAGExample]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"QASPER file not found at {p}. Convert QASPER to the universal "
            "JSONL schema before loading."
        )
    examples = load_jsonl(p)
    for ex in examples:
        if not ex.dataset:
            ex.dataset = "qasper"
    return examples

"""Loader for the universal JSONL format documented in the README."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from ..schemas import RAGExample


def load_jsonl(path: str | Path) -> list[RAGExample]:
    examples: list[RAGExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {exc}") from exc
            examples.append(RAGExample.from_dict(obj))
    return examples


def iter_jsonl(path: str | Path) -> Iterator[RAGExample]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield RAGExample.from_dict(json.loads(line))


def dump_jsonl(examples: Iterable[RAGExample], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(
                json.dumps(
                    {
                        "example_id": ex.example_id,
                        "dataset": ex.dataset,
                        "question": ex.question,
                        "answer": ex.answer,
                        "contexts": [
                            {"doc_id": c.doc_id, "title": c.title, "text": c.text}
                            for c in ex.contexts
                        ],
                        "gold_claim_labels": [
                            {
                                "claim_text": g.claim_text,
                                "label": g.label,
                                "support_doc_ids": g.support_doc_ids,
                            }
                            for g in ex.gold_claim_labels
                        ],
                        "is_synthetic": ex.is_synthetic,
                    }
                )
                + "\n"
            )
    return out

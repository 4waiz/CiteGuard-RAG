"""Generate synthetic perturbations from a clean RAG-example JSONL.

Outputs ``data/samples/perturbation_examples.jsonl`` (overwrites). Each
perturbation is clearly marked with ``is_synthetic: true`` and a
``perturbation`` tag so downstream analysis can group results by type.

Perturbations:
- drop_citations: remove every [doc...] marker from the answer.
- swap_citations: replace each citation with a different doc id from the example.
- fact_flip:      replace key tokens with a hard-coded antonym list (illustrative;
                  *not* a substitute for hand-written contradictions).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


CITATION_RE = re.compile(r"\[[^\]\[]+\]")

ANTONYMS = {
    "is": "is not",
    "was": "was not",
    "are": "are not",
    "first": "last",
    "largest": "smallest",
    "highest": "lowest",
    "boils": "freezes",
    "supports": "contradicts",
}


def _drop_citations(answer: str) -> str:
    return re.sub(r"\s+", " ", CITATION_RE.sub("", answer)).strip()


def _swap_citations(answer: str, doc_ids: list[str], rng: random.Random) -> str:
    if not doc_ids:
        return answer

    def swap(match: re.Match) -> str:
        new = rng.choice(doc_ids)
        return f"[{new}]"

    return CITATION_RE.sub(swap, answer)


def _fact_flip(answer: str) -> str:
    out = answer
    for word, repl in ANTONYMS.items():
        out = re.sub(rf"\b{re.escape(word)}\b", repl, out)
    return out


def perturb(example: dict, rng: random.Random) -> list[dict]:
    doc_ids = [c.get("doc_id", "") for c in example.get("contexts") or []]
    base_id = example.get("example_id", "ex")

    variants = [
        ("drop_citations", _drop_citations(example.get("answer", ""))),
        ("swap_citations", _swap_citations(example.get("answer", ""), doc_ids, rng)),
        ("fact_flip", _fact_flip(example.get("answer", ""))),
    ]
    out: list[dict] = []
    for tag, new_answer in variants:
        if new_answer == example.get("answer", ""):
            # Skip no-op perturbations.
            continue
        v = dict(example)
        v["example_id"] = f"{base_id}__{tag}"
        v["answer"] = new_answer
        v["is_synthetic"] = True
        v["perturbation"] = tag
        # We do NOT invent gold labels for the perturbation; drop them so the
        # downstream pipeline reports unsupervised diagnoses for these rows.
        v["gold_claim_labels"] = []
        out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/samples/custom_rag_examples.jsonl",
        help="Source JSONL of clean RAG examples.",
    )
    ap.add_argument(
        "--output",
        default="data/samples/perturbation_examples.jsonl",
        help="Where to write the synthetic perturbations.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        print(f"ERROR: source not found: {src}")
        return 1

    n_in = n_out = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "r", encoding="utf-8") as f_in, open(dst, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            n_in += 1
            for v in perturb(ex, rng):
                f_out.write(json.dumps(v, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"Read {n_in} examples; wrote {n_out} synthetic perturbations to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

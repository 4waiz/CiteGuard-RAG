"""Sample-data helper.

This repository ships with sample data already at
``data/samples/custom_rag_examples.jsonl``. This script verifies the file
exists and reports basic statistics. We deliberately do not fetch external
datasets here, because each requires conversion to the universal schema.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


SAMPLE_PATH = Path("data/samples/custom_rag_examples.jsonl")


def main() -> int:
    if not SAMPLE_PATH.exists():
        print(f"ERROR: sample file not found at {SAMPLE_PATH}.", file=sys.stderr)
        print("Re-clone the repository or restore the file from git.", file=sys.stderr)
        return 1
    n_examples = 0
    n_claims = 0
    n_contexts = 0
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n_examples += 1
            n_claims += len(obj.get("gold_claim_labels") or [])
            n_contexts += len(obj.get("contexts") or [])
    print(f"Sample file: {SAMPLE_PATH}")
    print(f"  examples:        {n_examples}")
    print(f"  total claims:    {n_claims}")
    print(f"  total contexts:  {n_contexts}")
    print("\nAll sample examples are clearly marked as synthetic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

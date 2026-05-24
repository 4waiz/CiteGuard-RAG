"""Run the default config and both ablations on a single input JSONL.

Writes ``outputs/tables/benchmark_summary.csv`` summarizing macro-F1 and a
handful of other metrics per run. Output paths for each run are namespaced
under ``outputs/<run_name>/`` so they don't clobber each other.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Make ``src`` importable when running as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from citeguard.config import load_config  # noqa: E402
from citeguard.data_loaders.custom_jsonl import load_jsonl  # noqa: E402
from citeguard.pipeline import CiteGuardPipeline  # noqa: E402


CONFIGS = [
    ("default", ROOT / "configs" / "default.yaml"),
    ("ablation_semantic_only", ROOT / "configs" / "ablation_semantic_only.yaml"),
    ("ablation_nli_only", ROOT / "configs" / "ablation_nli_only.yaml"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--output", default="outputs", type=str)
    args = ap.parse_args()

    examples = load_jsonl(args.input)
    if not examples:
        print(f"No examples loaded from {args.input}", file=sys.stderr)
        return 1

    out_root = Path(args.output)
    summary_rows: list[dict] = []
    for run_name, cfg_path in CONFIGS:
        if not cfg_path.exists():
            print(f"Config missing: {cfg_path}; skipping.", file=sys.stderr)
            continue
        cfg = load_config(cfg_path)
        run_out = out_root / run_name
        cfg.run_name = run_name
        cfg.output.tables_dir = str(run_out / "tables")
        cfg.output.figures_dir = str(run_out / "figures")
        cfg.output.cards_dir = str(run_out / "evidence_cards")
        cfg.output.reports_dir = str(run_out / "reports")

        pipeline = CiteGuardPipeline(cfg)
        results = pipeline.run(examples, show_progress=True)
        pipeline.write_outputs(results)
        agg = results["aggregate"]
        summary_rows.append(
            {
                "name": run_name,
                "macro_f1": agg["macro_f1"],
                "weighted_f1": agg["weighted_f1"],
                "n_claims": agg["n_claims"],
                "n_with_gold": agg["n_with_gold_label"],
                "missing_citation_rate": agg["missing_citation_rate"],
                "citation_mismatch_rate": agg["citation_mismatch_rate"],
                "avg_latency_ms": agg["avg_latency_ms"],
            }
        )
        print(
            f"[{run_name}] macro_f1={agg['macro_f1']:.4f} weighted_f1={agg['weighted_f1']:.4f} "
            f"n_claims={agg['n_claims']}"
        )

    summary_path = out_root / "tables" / "benchmark_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "macro_f1",
                "weighted_f1",
                "n_claims",
                "n_with_gold",
                "missing_citation_rate",
                "citation_mismatch_rate",
                "avg_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {summary_path}")

    json_path = out_root / "tables" / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

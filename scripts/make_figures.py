"""Generate all six required plots from existing CSV outputs.

Inputs (must already exist; usually written by ``citeguard evaluate``):
- outputs/tables/claim_eval.csv
- outputs/tables/aggregate_metrics.json (for confusion matrix)
- outputs/tables/benchmark_summary.csv (optional — for baseline/ablation plots)

Outputs (all six):
- outputs/figures/fig_error_distribution.png
- outputs/figures/fig_confusion_matrix.png
- outputs/figures/fig_baseline_comparison.png
- outputs/figures/fig_ablation.png
- outputs/figures/fig_retrieval_vs_support.png
- outputs/figures/fig_runtime.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from citeguard.reporting.plots import (  # noqa: E402
    plot_ablation,
    plot_baseline_comparison,
    plot_confusion_matrix,
    plot_error_distribution,
    plot_retrieval_vs_support,
    plot_runtime,
)
from citeguard.schemas import LABELS  # noqa: E402


def _load_confusion(agg_path: Path) -> tuple[np.ndarray, list[str]]:
    if not agg_path.exists():
        return np.zeros((len(LABELS), len(LABELS)), dtype=int), LABELS
    with open(agg_path, "r", encoding="utf-8") as f:
        agg = json.load(f)
    conf = agg.get("confusion", {})
    labels = conf.get("labels") or LABELS
    matrix = conf.get("matrix") or []
    if not matrix:
        return np.zeros((len(labels), len(labels)), dtype=int), labels
    return np.asarray(matrix, dtype=int), list(labels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim-eval", default="outputs/tables/claim_eval.csv")
    ap.add_argument("--aggregate", default="outputs/tables/aggregate_metrics.json")
    ap.add_argument("--benchmark", default="outputs/tables/benchmark_summary.csv")
    ap.add_argument("--figures-dir", default="outputs/figures")
    args = ap.parse_args()

    figs = Path(args.figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    claim_path = Path(args.claim_eval)
    if claim_path.exists():
        df = pd.read_csv(claim_path)
    else:
        print(f"WARNING: {claim_path} not found; producing empty figures.", file=sys.stderr)
        df = pd.DataFrame()

    plot_error_distribution(df, figs / "fig_error_distribution.png")

    cm, labels = _load_confusion(Path(args.aggregate))
    plot_confusion_matrix(cm, labels, figs / "fig_confusion_matrix.png")

    bench_path = Path(args.benchmark)
    if bench_path.exists():
        bench_df = pd.read_csv(bench_path)
        rows = bench_df.to_dict(orient="records")
    else:
        rows = []
    # Baseline comparison uses the same rows; the script user can supply more.
    plot_baseline_comparison(rows, figs / "fig_baseline_comparison.png")
    plot_ablation(rows, figs / "fig_ablation.png")

    plot_retrieval_vs_support(df, figs / "fig_retrieval_vs_support.png")
    plot_runtime(df, figs / "fig_runtime.png")

    for name in [
        "fig_error_distribution.png",
        "fig_confusion_matrix.png",
        "fig_baseline_comparison.png",
        "fig_ablation.png",
        "fig_retrieval_vs_support.png",
        "fig_runtime.png",
    ]:
        print(f"  {figs / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Regenerate summary tables from an existing claim_eval.csv.

Useful if you want to re-aggregate without re-running the pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from citeguard.metrics.classification import classification_report_dict, confusion_matrix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim-eval", default="outputs/tables/claim_eval.csv")
    ap.add_argument("--out", default="outputs/tables")
    args = ap.parse_args()

    path = Path(args.claim_eval)
    if not path.exists():
        print(f"ERROR: {path} not found. Run `citeguard evaluate` first.", file=sys.stderr)
        return 1
    df = pd.read_csv(path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_gold = df["gold_label"].notna() & (df["gold_label"].astype(str) != "")
    y_true = df.loc[has_gold, "gold_label"].astype(str).tolist()
    y_pred = df.loc[has_gold, "predicted_label"].astype(str).tolist()
    cls = classification_report_dict(y_true, y_pred)
    cm, labels = confusion_matrix(y_true, y_pred)

    label_dist = (
        df["predicted_label"].value_counts().rename_axis("label").reset_index(name="count")
    )
    label_dist.to_csv(out_dir / "label_distribution.csv", index=False)

    per_label_rows = [
        {"label": k, **v} for k, v in cls["per_label"].items()
    ]
    pd.DataFrame(per_label_rows).to_csv(out_dir / "per_label_metrics.csv", index=False)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    with open(out_dir / "classification_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {"macro_f1": cls["macro_f1"], "weighted_f1": cls["weighted_f1"], "labels": labels},
            f,
            indent=2,
        )
    print(f"Wrote label_distribution, per_label_metrics, confusion_matrix and classification_summary to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

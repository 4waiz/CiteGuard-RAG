"""Matplotlib plots used in the paper and dashboard.

All plots take in-memory data (no global state) and write a PNG. Functions
gracefully render an empty figure with an explanatory title when data is
empty, so a fresh install with no extra data still produces all six files.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..schemas import LABELS


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_error_distribution(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Bar chart of predicted_label counts."""
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    if df.empty or "predicted_label" not in df.columns:
        ax.set_title("Error distribution (no data)")
    else:
        counts = df["predicted_label"].value_counts().reindex(LABELS, fill_value=0)
        ax.bar(counts.index, counts.values, color="#4C72B0")
        ax.set_ylabel("Number of claims")
        ax.set_title("Predicted label distribution")
        ax.tick_params(axis="x", rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_confusion_matrix(
    matrix: np.ndarray, labels: list[str], out_path: str | Path
) -> Path:
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    if matrix.size == 0 or matrix.sum() == 0:
        ax.set_title("Confusion matrix (no gold labels)")
    else:
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, str(int(matrix[i, j])),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax)
        ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_baseline_comparison(
    rows: list[dict], out_path: str | Path
) -> Path:
    """Bar chart comparing macro-F1 across named systems.

    ``rows`` is a list of ``{"name": str, "macro_f1": float}``.
    """
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    if not rows:
        ax.set_title("Baseline comparison (no data)")
    else:
        names = [r["name"] for r in rows]
        vals = [float(r.get("macro_f1", 0.0)) for r in rows]
        ax.bar(names, vals, color="#55A868")
        ax.set_ylabel("Macro-F1")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Baseline comparison (macro-F1)")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_ablation(rows: list[dict], out_path: str | Path) -> Path:
    """Grouped bar chart of macro_f1 and weighted_f1 by config."""
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    if not rows:
        ax.set_title("Ablation (no data)")
    else:
        names = [r["name"] for r in rows]
        macro = [float(r.get("macro_f1", 0.0)) for r in rows]
        weighted = [float(r.get("weighted_f1", 0.0)) for r in rows]
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width / 2, macro, width, label="macro-F1", color="#4C72B0")
        ax.bar(x + width / 2, weighted, width, label="weighted-F1", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("F1")
        ax.set_title("Ablation comparison")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_retrieval_vs_support(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Scatter: semantic_score (x) vs entailment-derived support_score (y).

    Colored by predicted_label.
    """
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    if df.empty or "semantic_score" not in df.columns:
        ax.set_title("Retrieval vs support (no data)")
    else:
        cmap = plt.get_cmap("tab10")
        for i, lbl in enumerate(LABELS):
            sub = df[df["predicted_label"] == lbl]
            if sub.empty:
                continue
            ax.scatter(
                sub["semantic_score"],
                sub["support_score"],
                label=lbl,
                color=cmap(i % 10),
                s=30,
                alpha=0.75,
            )
        ax.set_xlabel("Semantic score (cosine)")
        ax.set_ylabel("Support score")
        ax.set_title("Retrieval similarity vs hybrid support score")
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_runtime(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Histogram of per-claim latency in milliseconds."""
    out = _ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    if df.empty or "latency_ms" not in df.columns:
        ax.set_title("Runtime per claim (no data)")
    else:
        ax.hist(df["latency_ms"], bins=20, color="#8172B3")
        ax.set_xlabel("Latency per claim (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Per-claim runtime distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

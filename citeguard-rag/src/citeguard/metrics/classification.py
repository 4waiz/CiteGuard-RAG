"""Classification metrics for predicted vs gold claim labels."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from ..schemas import LABELS


def classification_report_dict(
    y_true: list[str],
    y_pred: list[str],
    labels: Iterable[str] | None = None,
) -> dict:
    """Return a dict with macro/weighted F1 and per-label P/R/F1.

    Empty inputs return zeros so callers don't have to special-case the
    "no gold labels in this run" path.
    """
    labels = list(labels) if labels is not None else LABELS
    if not y_true:
        return {
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_label": {lbl: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for lbl in labels},
            "sklearn_report": "",
        }
    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_label = {
        lbl: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(s[i]),
        }
        for i, lbl in enumerate(labels)
    }
    return {
        "macro_f1": macro,
        "weighted_f1": weighted,
        "per_label": per_label,
        "sklearn_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
    }


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return (matrix, label_order)."""
    labels = list(labels) if labels is not None else LABELS
    if not y_true:
        return np.zeros((len(labels), len(labels)), dtype=int), labels
    return sk_confusion_matrix(y_true, y_pred, labels=labels), labels

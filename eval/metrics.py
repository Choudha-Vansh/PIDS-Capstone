"""
Metrics for the intent model, including calibration (ECE). Kept dependency-light:
sklearn for the standard curves, numpy for ECE.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 10) -> float:
    """
    ECE: weighted average gap between confidence and accuracy across bins.
    A well-calibrated model has ECE near 0.
    """
    probs = np.asarray(probs); labels = np.asarray(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (probs > lo) & (probs <= hi) if b > 0 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def classification_report(probs: np.ndarray, labels: np.ndarray,
                          threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)
    out = {
        "accuracy": float((preds == labels).mean()),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "ece": expected_calibration_error(probs, labels),
    }
    # AUC / AP need both classes present
    if len(np.unique(labels)) > 1:
        out["auc"] = float(roc_auc_score(labels, probs))
        out["ap"] = float(average_precision_score(labels, probs))
    else:
        out["auc"] = float("nan")
        out["ap"] = float("nan")
    return out

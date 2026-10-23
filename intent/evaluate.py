"""
Evaluate the trained intent model on the held-out TEST split (never used for model
selection or calibration). Applies temperature calibration if available. Writes
metrics + plots to results/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from intent.dataset import SequenceDataset          # noqa: E402
from intent.model import build_model                # noqa: E402
from eval.metrics import classification_report      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    art = cfg["paths"]["artifacts"]; wd = cfg["paths"]["weights"]
    rd = cfg["paths"]["results"]; os.makedirs(rd, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SequenceDataset(f"{art}/X_test.npy", f"{art}/y_test.npy")
    dl = DataLoader(ds, batch_size=256)
    model = build_model(ds.feature_dim, cfg).to(device)
    model.load_state_dict(torch.load(f"{wd}/intent_model.pth", map_location=device))
    model.eval()

    T = 1.0
    cal = f"{wd}/calibration.json"
    if os.path.exists(cal):
        T = json.load(open(cal))["temperature"]
        print(f"Using temperature T={T:.3f}")

    logits, labels = [], []
    with torch.no_grad():
        for X, y in dl:
            logits.append(model(X.to(device)).cpu().numpy())
            labels.append(y.numpy())
    logits = np.concatenate(logits); labels = np.concatenate(labels)
    probs = 1.0 / (1.0 + np.exp(-logits / T))

    rep = classification_report(probs, labels)
    print("\n=== TEST metrics ===")
    for k, v in rep.items():
        print(f"  {k:10s}: {v:.4f}")
    json.dump(rep, open(f"{rd}/test_metrics.json", "w"), indent=2)

    _plots(probs, labels, rd)
    print(f"\nSaved metrics + plots -> {rd}/")


def _plots(probs, labels, rd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (confusion_matrix, roc_curve,
                                 precision_recall_curve)

    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels.astype(int), preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Not", "Cross"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Not", "Cross"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(f"{rd}/confusion_matrix.png", dpi=150); plt.close()

    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure(figsize=(4, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
        plt.tight_layout(); plt.savefig(f"{rd}/roc_curve.png", dpi=150); plt.close()

        pr, rc, _ = precision_recall_curve(labels, probs)
        plt.figure(figsize=(4, 4)); plt.plot(rc, pr)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve")
        plt.tight_layout(); plt.savefig(f"{rd}/precision_recall.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()

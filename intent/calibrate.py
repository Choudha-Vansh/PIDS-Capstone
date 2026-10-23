"""
Temperature scaling: fit a single scalar T on the validation set so that
sigmoid(logit / T) gives calibrated probabilities. Saves T to weights/calibration.json.
Calibrated probabilities matter here because the AEB gate is a probability threshold —
an over-confident model would brake too eagerly or too late.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from intent.dataset import SequenceDataset          # noqa: E402
from intent.model import build_model                # noqa: E402
from eval.metrics import expected_calibration_error  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    art = cfg["paths"]["artifacts"]; wd = cfg["paths"]["weights"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SequenceDataset(f"{art}/X_val.npy", f"{art}/y_val.npy")
    dl = DataLoader(ds, batch_size=256)
    model = build_model(ds.feature_dim, cfg).to(device)
    model.load_state_dict(torch.load(f"{wd}/intent_model.pth", map_location=device))
    model.eval()

    logits, labels = [], []
    with torch.no_grad():
        for X, y in dl:
            logits.append(model(X.to(device)).cpu())
            labels.append(y)
    logits = torch.cat(logits); labels = torch.cat(labels)

    T = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    T_val = float(T.detach().item())

    probs_before = torch.sigmoid(logits).numpy()
    probs_after = torch.sigmoid(logits / T_val).numpy()
    lbl = labels.numpy()
    ece_b = expected_calibration_error(probs_before, lbl)
    ece_a = expected_calibration_error(probs_after, lbl)

    json.dump({"temperature": T_val}, open(f"{wd}/calibration.json", "w"))
    print(f"Temperature = {T_val:.3f}")
    print(f"ECE before {ece_b:.4f}  ->  after {ece_a:.4f}")
    print(f"Saved -> {wd}/calibration.json")


if __name__ == "__main__":
    main()

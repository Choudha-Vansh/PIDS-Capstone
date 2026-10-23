"""
Train the temporal crossing-intent model.

    python intent/train.py [--config config/config.yaml]

Reads artifacts/{X,y}_{train,val}.npy (produced by data/combine.py), trains with
class-weighted BCEWithLogitsLoss, early-stops on val loss, saves the best weights to
weights/intent_model.pth plus a small meta json (feature dim, seq length).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    art = cfg["paths"]["artifacts"]
    weights_dir = cfg["paths"]["weights"]
    os.makedirs(weights_dir, exist_ok=True)

    tc = cfg["train"]
    torch.manual_seed(tc["seed"]); np.random.seed(tc["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = SequenceDataset(f"{art}/X_train.npy", f"{art}/y_train.npy")
    val_ds = SequenceDataset(f"{art}/X_val.npy", f"{art}/y_val.npy")
    in_dim = train_ds.feature_dim
    print(f"Train {len(train_ds)}  Val {len(val_ds)}  feature_dim={in_dim}")

    train_dl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=tc["batch_size"])

    # class weight for imbalance -> pos_weight for BCEWithLogitsLoss
    pos = float(train_ds.y.sum()); neg = float(len(train_ds.y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    print(f"Class balance -> crossing {int(pos)}  not {int(neg)}  "
          f"pos_weight={pos_weight.item():.2f}")

    model = build_model(in_dim, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"],
                           weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=4,
                                                       factor=0.5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val, patience, bad = float("inf"), tc["early_stop_patience"], 0
    for epoch in range(1, tc["epochs"] + 1):
        model.train(); tl = 0.0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(X), y)
            loss.backward(); opt.step()
            tl += loss.item()
        tl /= len(train_dl)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                vl += criterion(model(X), y).item()
        vl /= max(len(val_dl), 1)
        sched.step(vl)
        print(f"Epoch {epoch:02d}  train {tl:.4f}  val {vl:.4f}")

        if vl < best_val:
            best_val = vl; bad = 0
            torch.save(model.state_dict(), f"{weights_dir}/intent_model.pth")
            json.dump({"in_dim": in_dim,
                       "seq_length": cfg["intent"]["seq_length"],
                       "model_type": cfg["intent"]["model_type"]},
                      open(f"{weights_dir}/intent_meta.json", "w"))
            print("   ↑ saved best")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    print(f"Done. Best val loss {best_val:.4f} -> {weights_dir}/intent_model.pth")


if __name__ == "__main__":
    main()

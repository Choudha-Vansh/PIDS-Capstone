"""
Combine PIE + JAAD sequence sets and split into train/val/test BY CLIP, so frames
from the same video never leak across splits (a common way to accidentally inflate
metrics). Writes artifacts/{X,y}_{train,val,test}.npy.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml


def _load(path):
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"], d["clip_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    art = cfg["paths"]["artifacts"]
    split = cfg["train"]["split"]
    seed = cfg["train"]["seed"]

    parts = []
    for name in ("pie_sequences.npz", "jaad_sequences.npz"):
        loaded = _load(os.path.join(art, name))
        if loaded is None:
            print(f"skip missing {name}")
            continue
        X, y, cids = loaded
        parts.append((X, y, np.array([f"{name[:3]}::{c}" for c in cids])))
        print(f"{name}: {X.shape}")

    if not parts:
        print("ERROR: no prepared sequences found. Run prepare_pie.py / prepare_jaad.py.")
        sys.exit(1)

    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    cids = np.concatenate([p[2] for p in parts])
    print(f"Combined: X={X.shape}  crossing={int(y.sum())}/{len(y)}")

    # split by unique clip id
    rng = np.random.RandomState(seed)
    clips = np.array(sorted(set(cids.tolist())))
    rng.shuffle(clips)
    n = len(clips)
    n_tr = int(split["train"] * n); n_va = int(split["val"] * n)
    train_c = set(clips[:n_tr]); val_c = set(clips[n_tr:n_tr + n_va])
    test_c = set(clips[n_tr + n_va:])

    def mask(cset):
        return np.array([c in cset for c in cids])

    for split_name, cset in (("train", train_c), ("val", val_c), ("test", test_c)):
        m = mask(cset)
        np.save(os.path.join(art, f"X_{split_name}.npy"), X[m])
        np.save(os.path.join(art, f"y_{split_name}.npy"), y[m])
        print(f"{split_name}: {int(m.sum())} sequences from {len(cset)} clips  "
              f"(crossing {int(y[m].sum())})")

    print("Done. Splits are clip-disjoint (no leakage).")


if __name__ == "__main__":
    main()

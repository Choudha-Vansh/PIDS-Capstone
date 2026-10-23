"""
Online intent inference for the live/demo pipeline. Maintains a rolling feature
history per track and only predicts once a FULL real sequence is available (no
zero-padding) — the same rule used to build training windows, so train and inference
distributions match. Applies temperature calibration if present.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from typing import Dict, Optional

import numpy as np


class OnlineIntent:
    def __init__(self, cfg: dict, device=None):
        import torch
        from intent.model import build_model
        self.torch = torch
        self.seq_len = cfg["intent"]["seq_length"]
        wd = cfg["paths"]["weights"]
        meta = json.load(open(f"{wd}/intent_meta.json"))
        self.in_dim = meta["in_dim"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.in_dim, cfg).to(self.device)
        self.model.load_state_dict(
            torch.load(f"{wd}/intent_model.pth", map_location=self.device))
        self.model.eval()
        self.T = 1.0
        cal = f"{wd}/calibration.json"
        if os.path.exists(cal):
            self.T = json.load(open(cal))["temperature"]
        self.hist: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.seq_len))

    def update(self, track_id: int, feature_vec: np.ndarray) -> Optional[float]:
        """
        Push one frame's feature vector for a track; return calibrated crossing
        probability once a full window exists, else None.
        """
        self.hist[track_id].append(feature_vec.astype(np.float32))
        if len(self.hist[track_id]) < self.seq_len:
            return None
        seq = np.stack(self.hist[track_id])[None]           # (1, T, F)
        with self.torch.no_grad():
            logit = self.model(self.torch.from_numpy(seq).to(self.device)).item()
        return 1.0 / (1.0 + np.exp(-logit / self.T))

    def drop(self, track_id: int) -> None:
        self.hist.pop(track_id, None)

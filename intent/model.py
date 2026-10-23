"""
Temporal crossing-intent model. Two backbones selectable by config:
  - "gru": bidirectional? no — causal GRU (works for online inference) + attention pool
  - "transformer": small encoder over the sequence + attention pool

Outputs a single RAW LOGIT per sequence (apply torch.sigmoid at inference). Training
uses BCEWithLogitsLoss(pos_weight=...) so class imbalance is handled properly.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Learned weighted average over the time dimension."""
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):                      # x: (B, T, D)
        w = torch.softmax(self.score(x), dim=1)  # (B, T, 1)
        return (w * x).sum(dim=1)              # (B, D)


class GRUIntent(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.pool = AttentionPool(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):                      # (B, T, F)
        h, _ = self.gru(x)                     # (B, T, H)
        pooled = self.pool(h)                  # (B, H)
        return self.head(pooled).squeeze(1)    # (B,) raw logits


class TransformerIntent(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, dropout=0.3, nhead=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        enc = nn.TransformerEncoderLayer(hidden, nhead, hidden * 2, dropout,
                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.pool = AttentionPool(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.encoder(self.proj(x))
        return self.head(self.pool(h)).squeeze(1)


def build_model(in_dim: int, cfg: dict) -> nn.Module:
    ic = cfg.get("intent", {})
    mtype = ic.get("model_type", "gru")
    hidden = ic.get("hidden_size", 128)
    layers = ic.get("num_layers", 2)
    dropout = ic.get("dropout", 0.3)
    if mtype == "transformer":
        return TransformerIntent(in_dim, hidden, layers, dropout)
    return GRUIntent(in_dim, hidden, layers, dropout)

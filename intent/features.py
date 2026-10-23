"""
Builds the per-frame feature vector fed to the temporal intent model, and slices
per-track histories into fixed-length sequences with a future crossing label.

Feature vector (config-driven; see FEATURE_LAYOUT):
    normalized keypoints (17 * 2 = 34)   [if use_pose]
    bbox normalized (4)                  xc, yc, w, h in [0,1]
    3D position (3)                      x, y, z meters (0 on camera path w/o fusion)
    3D velocity (3)                      vx, vy, vz m/s
    dist_to_ego_path (1)                 lateral offset outside corridor (m)
    ego_speed (1)                        m/s

The same builder is used offline (dataset prep) and online (live dashboard), so
training and inference see identical features — the mismatch that hurt the earlier
prototype cannot happen here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

N_KEYPOINTS = 17


@dataclass
class FeatureConfig:
    use_pose: bool = True
    use_3d: bool = True

    @property
    def dim(self) -> int:
        d = 0
        if self.use_pose:
            d += N_KEYPOINTS * 2
        d += 4                       # bbox
        if self.use_3d:
            d += 3 + 3 + 1           # pos, vel, dist_to_path
        d += 1                       # ego speed
        return d


def build_frame_feature(
    bbox_xywh_norm: np.ndarray,        # (4,)
    keypoints_norm: Optional[np.ndarray],   # (17,2) normalized to [0,1] or None
    pos_3d: Optional[np.ndarray],      # (3,) or None
    vel_3d: Optional[np.ndarray],      # (3,) or None
    dist_to_path_m: float,
    ego_speed_mps: float,
    cfg: FeatureConfig,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    if cfg.use_pose:
        kp = (keypoints_norm if keypoints_norm is not None
              else np.zeros((N_KEYPOINTS, 2), np.float32))
        parts.append(kp.reshape(-1).astype(np.float32))
    parts.append(np.asarray(bbox_xywh_norm, np.float32))
    if cfg.use_3d:
        p = pos_3d if pos_3d is not None else np.zeros(3, np.float32)
        v = vel_3d if vel_3d is not None else np.zeros(3, np.float32)
        # clip inf (no-range) to a large finite sentinel so training is stable
        p = np.where(np.isfinite(p), p, 0.0).astype(np.float32)
        v = np.where(np.isfinite(v), v, 0.0).astype(np.float32)
        parts.append(p)
        parts.append(v)
        parts.append(np.array([dist_to_path_m], np.float32))
    parts.append(np.array([ego_speed_mps], np.float32))
    return np.concatenate(parts).astype(np.float32)


def normalize_keypoints(kpts_px: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """(17,3) pixel keypoints -> (17,2) normalized xy in [0,1]."""
    out = np.zeros((N_KEYPOINTS, 2), np.float32)
    out[:, 0] = kpts_px[:, 0] / max(img_w, 1)
    out[:, 1] = kpts_px[:, 1] / max(img_h, 1)
    return out


def slice_sequences(
    frame_features: np.ndarray,        # (T, F)
    labels: np.ndarray,                # (T,) crossing 0/1 per frame
    frame_numbers: np.ndarray,         # (T,) real frame indices (gap-aware)
    seq_length: int,
    predict_horizon: int,
):
    """
    Yield (window (seq_length, F), label) where label is the crossing state
    `predict_horizon` frames after the window end. Windows that span a frame gap
    (dropped/occluded frames) are skipped so we never learn across a discontinuity.
    """
    X, y = [], []
    T = len(frame_features)
    end = T - predict_horizon
    for i in range(end - seq_length):
        win_frames = frame_numbers[i:i + seq_length]
        # require the window + horizon to be contiguous in real frame numbers
        target_idx = i + seq_length + predict_horizon - 1
        if target_idx >= T:
            continue
        span = frame_numbers[target_idx] - win_frames[0]
        if span != (seq_length + predict_horizon - 1):
            continue   # gap inside window/horizon -> skip
        X.append(frame_features[i:i + seq_length])
        y.append(labels[target_idx])
    return X, y

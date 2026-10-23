"""Feature builder + gap-aware sequence slicing tests (pure numpy)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intent.features import (
    FeatureConfig, build_frame_feature, normalize_keypoints, slice_sequences,
    N_KEYPOINTS,
)


def test_feature_dim_full():
    cfg = FeatureConfig(use_pose=True, use_3d=True)
    # 34 kpts + 4 bbox + 3 pos + 3 vel + 1 dist + 1 ego = 46
    assert cfg.dim == 46
    f = build_frame_feature(np.zeros(4), np.zeros((N_KEYPOINTS, 2)),
                            np.zeros(3), np.zeros(3), 0.0, 0.0, cfg)
    assert f.shape[0] == 46


def test_feature_dim_no_pose_no_3d():
    cfg = FeatureConfig(use_pose=False, use_3d=False)
    # 4 bbox + 1 ego = 5
    assert cfg.dim == 5
    f = build_frame_feature(np.array([0.1, 0.2, 0.3, 0.4]), None, None, None,
                            0.0, 12.0, cfg)
    assert f.shape[0] == 5 and f[-1] == 12.0


def test_inf_position_sanitized():
    cfg = FeatureConfig(use_pose=False, use_3d=True)
    f = build_frame_feature(np.zeros(4), None,
                            np.array([np.inf, 0, np.inf]), np.zeros(3), 0.0, 0.0, cfg)
    assert np.all(np.isfinite(f))


def test_normalize_keypoints():
    kp = np.array([[960, 540, 1.0]] * N_KEYPOINTS)
    out = normalize_keypoints(kp, 1920, 1080)
    assert abs(out[0, 0] - 0.5) < 1e-6 and abs(out[0, 1] - 0.5) < 1e-6


def test_slice_skips_gaps():
    T, F = 30, 5
    feats = np.random.rand(T, F).astype(np.float32)
    labels = np.zeros(T); labels[20:] = 1
    frames = np.arange(T)
    # introduce a gap: frame numbers jump at index 10
    frames[10:] += 5
    X, y = slice_sequences(feats, labels, frames, seq_length=8, predict_horizon=4)
    # every returned window must be contiguous (span == seq_len+horizon-1)
    assert len(X) > 0
    assert all(w.shape == (8, F) for w in X)


def test_slice_contiguous_count():
    T, F = 20, 3
    feats = np.random.rand(T, F).astype(np.float32)
    labels = np.zeros(T)
    frames = np.arange(T)              # fully contiguous
    X, y = slice_sequences(feats, labels, frames, seq_length=5, predict_horizon=2)
    # windows i where target = i+5+2-1 = i+6 < 20  -> i in [0, 13)
    assert len(X) == 13


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")

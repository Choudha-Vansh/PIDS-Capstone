"""Kalman filter sanity: it should recover a constant velocity from noisy positions."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perception.kalman import KalmanCV3D


def test_recovers_velocity():
    rng = np.random.RandomState(0)
    dt = 0.1
    kf = KalmanCV3D(dt=dt, process_var=0.01, meas_var=0.05)
    true_v = np.array([0.5, 0.0, -2.0])   # approaching along +forward (z decreasing)
    pos = np.array([0.0, 0.0, 30.0])
    for _ in range(80):
        pos = pos + true_v * dt
        meas = pos + rng.normal(0, 0.1, size=3)
        kf.update(meas)
    est_v = kf.velocity
    assert np.allclose(est_v, true_v, atol=0.3), f"velocity off: {est_v} vs {true_v}"


def test_position_tracks():
    kf = KalmanCV3D(dt=0.1)
    kf.update(np.array([1.0, 2.0, 10.0]))
    p = kf.position
    assert abs(p[2] - 10.0) < 2.0


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

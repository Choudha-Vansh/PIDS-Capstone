"""
Constant-velocity Kalman filter for smoothing a pedestrian's 3D position and
estimating velocity. State = [x, y, z, vx, vy, vz]. Pure numpy (testable, no torch).
"""
from __future__ import annotations

import numpy as np


class KalmanCV3D:
    def __init__(self, dt: float = 1.0 / 30.0,
                 process_var: float = 1.0, meas_var: float = 0.3):
        self.dt = dt
        self.x = np.zeros(6)             # state
        self.P = np.eye(6) * 10.0        # covariance
        self.initialized = False

        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i + 3] = dt

        self.H = np.zeros((3, 6))        # we measure position only
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

        self.Q = np.eye(6) * process_var
        self.R = np.eye(3) * meas_var

    def init_state(self, pos_xyz: np.ndarray) -> None:
        self.x[:3] = pos_xyz
        self.x[3:] = 0.0
        self.initialized = True

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def update(self, pos_xyz: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.init_state(pos_xyz)
            return self.x[:3].copy()
        self.predict()
        z = np.asarray(pos_xyz, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3].copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

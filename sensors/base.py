"""
Sensor input abstraction. This is the seam that keeps the project hardware-portable:
today the frames come from a video file or nuScenes; later they come from a real
camera/radar/LIDAR rig on a Jetson. Everything downstream depends only on SensorFrame
and SensorSource, never on where the data came from.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np


@dataclass
class RadarReturn:
    """One radar detection in ego frame (meters, m/s)."""
    x: float
    y: float
    range_m: float
    radial_velocity_mps: float


@dataclass
class SensorFrame:
    """
    A single synchronized time step from whatever sensors are available.
    The camera image is always present; radar/lidar are optional (None on the
    camera-only demo-video path).
    """
    frame_idx: int
    timestamp_s: float
    image: np.ndarray                              # HxWx3 BGR
    ego_speed_mps: float = 0.0                     # 0 if unknown (e.g. dashcam)
    radar: Optional[list] = None                   # list[RadarReturn]
    lidar_points: Optional[np.ndarray] = None      # Nx3 (x,y,z) ego frame
    tof_distance_m: Optional[float] = None         # front ToF range (m), None if absent
    calib: dict = field(default_factory=dict)      # intrinsics/extrinsics if known

    @property
    def has_range_sensors(self) -> bool:
        return self.radar is not None or self.lidar_points is not None


class SensorSource(ABC):
    """Iterable source of SensorFrames."""

    @abstractmethod
    def __iter__(self) -> Iterator[SensorFrame]:
        ...

    @property
    @abstractmethod
    def fps(self) -> float:
        ...

    @property
    def total_frames(self) -> Optional[int]:
        return None

    def close(self) -> None:
        pass

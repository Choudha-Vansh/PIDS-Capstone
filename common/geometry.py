"""
Geometry helpers. Monocular distance estimation for the camera-only (demo-video)
path, plus small vector utilities. All distances in meters, angles in radians.
"""
from __future__ import annotations

import math
from typing import Tuple


def monocular_distance_from_height(
    bbox_height_px: float,
    focal_px: float,
    assumed_ped_height_m: float = 1.7,
) -> float:
    """
    Pinhole distance estimate from apparent pedestrian height:
        distance = (real_height * focal_length_px) / apparent_height_px

    This is a rough estimate — accuracy depends on focal_px calibration and on
    the pedestrian actually being ~assumed height and fully visible. On real
    sensors this is replaced by radar/LIDAR range (see perception/fusion.py).
    """
    if bbox_height_px <= 1.0:
        return float("inf")
    return (assumed_ped_height_m * focal_px) / bbox_height_px


def closing_speed_mps(
    prev_distance_m: float,
    curr_distance_m: float,
    dt_s: float,
) -> float:
    """
    Positive closing speed = gap is shrinking (pedestrian getting closer / ego
    approaching). Returns 0 for opening gaps so TTC stays well-defined.
    """
    if dt_s <= 0:
        return 0.0
    rate = (prev_distance_m - curr_distance_m) / dt_s
    return max(0.0, rate)


def lateral_offset_to_path(
    ped_x_m: float,
    ego_path_half_width_m: float = 1.2,
) -> float:
    """
    How far outside the ego travel corridor the pedestrian is (meters).
    0 means inside the corridor (in the vehicle's path). Used to gate AEB so we
    don't brake for pedestrians safely off to the side.
    """
    return max(0.0, abs(ped_x_m) - ego_path_half_width_m)


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

"""
Camera-only 3D localization for the demo-video path. Without radar/LIDAR we can't
measure true depth, so we estimate it from apparent pedestrian height (pinhole) and
place the pedestrian on the ground plane in front of the ego vehicle.

Longitudinal distance (z, forward) comes from the height method; lateral offset (x)
comes from horizontal image position scaled by that distance. This is approximate —
the fusion path (perception/fusion.py) replaces it with metric range on real sensors.
"""
from __future__ import annotations

import numpy as np

from common.geometry import monocular_distance_from_height
from perception.detector import Detection


def localize_monocular(det: Detection, image_w: int, focal_px: float,
                       assumed_ped_height_m: float = 1.7) -> np.ndarray:
    """
    Return an approximate ego-frame position [x_lateral, y_up, z_forward] in meters.
    """
    z = monocular_distance_from_height(det.bbox_height_px, focal_px,
                                       assumed_ped_height_m)
    cx, _ = det.center_px
    # horizontal angle from image center -> lateral offset at distance z
    x = (cx - image_w / 2.0) / focal_px * z
    return np.array([x, 0.0, z], dtype=float)

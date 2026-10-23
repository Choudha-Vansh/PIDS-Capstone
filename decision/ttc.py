"""
Time-To-Collision. TTC = longitudinal distance / closing speed.
Kept deliberately simple and unit-tested; the decision policy consumes it.
"""
from __future__ import annotations

import math


def time_to_collision(distance_m: float, closing_speed_mps: float) -> float:
    """
    Returns seconds until collision, or +inf if the gap isn't closing.
    """
    if closing_speed_mps <= 1e-6:
        return math.inf
    if distance_m <= 0:
        return 0.0
    return distance_m / closing_speed_mps


def is_in_conflict(lateral_offset_m: float, corridor_margin_m: float = 0.3) -> bool:
    """
    True if the pedestrian is inside (or within a margin of) the ego corridor,
    i.e. actually in the vehicle's path. lateral_offset_m comes from
    common.geometry.lateral_offset_to_path (0 == inside corridor).
    """
    return lateral_offset_m <= corridor_margin_m

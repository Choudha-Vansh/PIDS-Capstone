"""
ToF range safety net — an INDEPENDENT fallback that stops the Duckiebot before a hard
distance, regardless of what the camera/pedestrian path decided.

Why separate from the camera hazard logic:
  - The camera path (detect -> intent -> cruise policy) is the "smart" layer: it decides
    to brake for a pedestrian in the bot's path.
  - This layer is the "dumb but reliable" last line of defense: if that path fails to
    stop in time (missed detection, latency, mis-classification) OR the obstacle isn't a
    pedestrian at all, the ToF range forces a stop once anything is within
    `safety_stop_distance_m`.
  - It OVERRIDES the camera command (can only make the bot slower/stopped, never faster),
    so it is strictly a safety addition.

Pure Python (no ROS), unit-tested. The ROS subscriber lives in sensors/tof_range.py and
hands this layer a distance in metres (or None).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from control.cruise_policy import DriveCommand, DriveState


@dataclass
class SafetyConfig:
    enabled: bool = True
    safety_stop_distance_m: float = 0.30
    max_valid_m: float = 1.20
    unit: str = "m"

    @classmethod
    def from_config(cls, cfg: dict) -> "SafetyConfig":
        t = cfg.get("duckiebot", {}).get("tof", {})
        return cls(
            enabled=t.get("enabled", True),
            safety_stop_distance_m=t.get("safety_stop_distance_m", 0.30),
            max_valid_m=t.get("max_valid_m", 1.20),
            unit=t.get("unit", "m"),
        )


def normalize_range(value: Optional[float], unit: str) -> Optional[float]:
    """Convert a raw ToF reading to metres. Accepts 'm' or 'cm'."""
    if value is None:
        return None
    return value / 100.0 if unit == "cm" else float(value)


def is_valid(range_m: Optional[float], cfg: SafetyConfig) -> bool:
    """A usable reading: present, positive, and within the sensor's trusted range."""
    return range_m is not None and 0.0 < range_m <= cfg.max_valid_m


def tof_triggered(range_m: Optional[float], cfg: SafetyConfig) -> bool:
    """True when the ToF says STOP (valid reading at/under the hard threshold)."""
    if not cfg.enabled or not is_valid(range_m, cfg):
        return False
    return range_m <= cfg.safety_stop_distance_m


def apply_safety_stop(cmd: DriveCommand, range_m: Optional[float],
                      cfg: SafetyConfig) -> DriveCommand:
    """
    Fuse the camera-derived drive command with the ToF safety net. If the ToF has
    tripped, force a stop (and tag the reason); otherwise pass the command through.
    """
    if tof_triggered(range_m, cfg):
        cm = range_m * 100.0
        return DriveCommand(0.0, 0.0, DriveState.STOPPED, braking=True,
                            reason=f"ToF SAFETY STOP (object @ {cm:.0f} cm)")
    return cmd

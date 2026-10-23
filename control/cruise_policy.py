"""
Cruise / brake / resume state machine for the Duckiebot live loop.

Behaviour (per the hardware bring-up spec):
    - default: CRUISE forward at cruise_speed
    - a pedestrian hazard in the path -> BRAKE to a stop
    - path clear again -> RESUME cruising

Debounce + hysteresis keep the motors from chattering on a flickering detection:
    - brake only after `brake_on_frames` consecutive hazard frames
    - resume only after `resume_clear_frames` consecutive clear frames

Pure Python, no ROS / torch — unit-tested. The live loop feeds it a single boolean
`hazard` per frame and reads back the commanded (v, omega).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DriveState(Enum):
    CRUISE = "CRUISE"
    STOPPED = "STOPPED"


@dataclass
class CruiseConfig:
    cruise_speed_mps: float = 0.22
    cruise_omega: float = 0.0
    brake_on_frames: int = 3
    resume_clear_frames: int = 8

    @classmethod
    def from_config(cls, cfg: dict) -> "CruiseConfig":
        c = cfg.get("duckiebot", {}).get("control", {})
        return cls(
            cruise_speed_mps=c.get("cruise_speed_mps", 0.22),
            cruise_omega=c.get("cruise_omega", 0.0),
            brake_on_frames=c.get("brake_on_frames", 3),
            resume_clear_frames=c.get("resume_clear_frames", 8),
        )


@dataclass
class DriveCommand:
    v: float            # linear speed (m/s); 0 = stopped/braking
    omega: float        # angular speed (rad/s)
    state: DriveState
    braking: bool       # True on the frame(s) a brake is being commanded
    reason: str


class CruisePolicy:
    """Feed it `hazard` (bool) each frame; it returns the DriveCommand to send."""

    def __init__(self, cfg: CruiseConfig):
        self.cfg = cfg
        self.state = DriveState.CRUISE
        self._hazard_streak = 0
        self._clear_streak = 0

    def step(self, hazard: bool) -> DriveCommand:
        if hazard:
            self._hazard_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            self._hazard_streak = 0

        if self.state == DriveState.CRUISE:
            if self._hazard_streak >= self.cfg.brake_on_frames:
                self.state = DriveState.STOPPED
                return DriveCommand(0.0, 0.0, self.state, braking=True,
                                    reason="pedestrian in path -> BRAKE")
            return DriveCommand(self.cfg.cruise_speed_mps, self.cfg.cruise_omega,
                                self.state, braking=False, reason="path clear -> cruise")

        # STOPPED
        if self._clear_streak >= self.cfg.resume_clear_frames:
            self.state = DriveState.CRUISE
            return DriveCommand(self.cfg.cruise_speed_mps, self.cfg.cruise_omega,
                                self.state, braking=False, reason="path clear -> ACCELERATE")
        return DriveCommand(0.0, 0.0, self.state, braking=True,
                            reason="holding stop (pedestrian / not yet clear)")

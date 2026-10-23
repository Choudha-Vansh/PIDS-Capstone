"""
Simple longitudinal vehicle-dynamics stopping model (simulated ego vehicle).

Given ego speed, distance to a pedestrian in the path, and the AEB level, compute:
  - stopping distance (reaction/actuation delay coast + braking distance)
  - whether a collision is avoided
  - final speed at the pedestrian's position (impact speed if not avoided)

This is a point-mass model with constant deceleration — enough to reason about
whether the AEB decision would have prevented a collision in the simulation.
"""
from __future__ import annotations

from dataclasses import dataclass

from decision.aeb_policy import AEBLevel


@dataclass
class VehicleParams:
    mass_kg: float = 1500.0
    max_decel_mps2: float = 8.0
    partial_decel_mps2: float = 3.5
    actuation_delay_s: float = 0.2
    road_friction: float = 0.8

    @classmethod
    def from_config(cls, cfg: dict) -> "VehicleParams":
        v = cfg.get("vehicle", {})
        return cls(
            mass_kg=v.get("mass_kg", 1500.0),
            max_decel_mps2=v.get("max_decel_mps2", 8.0),
            partial_decel_mps2=v.get("partial_decel_mps2", 3.5),
            actuation_delay_s=v.get("actuation_delay_s", 0.2),
            road_friction=v.get("road_friction", 0.8),
        )


@dataclass
class StoppingResult:
    decel_mps2: float
    coast_distance_m: float      # distance covered during actuation delay
    braking_distance_m: float    # distance covered while braking to stop
    stopping_distance_m: float   # total
    collision_avoided: bool
    impact_speed_mps: float      # 0 if avoided


def _decel_for_level(level: AEBLevel, p: VehicleParams) -> float:
    if level >= AEBLevel.FULL_AEB:
        return min(p.max_decel_mps2, p.road_friction * 9.81)
    if level == AEBLevel.PARTIAL_AEB:
        return p.partial_decel_mps2
    return 0.0


def simulate_stop(
    ego_speed_mps: float,
    distance_to_ped_m: float,
    level: AEBLevel,
    params: VehicleParams,
) -> StoppingResult:
    """Point-mass constant-deceleration stop with an actuation delay coast phase."""
    decel = _decel_for_level(level, params)

    if decel <= 0.0:
        # No braking applied: vehicle covers the whole gap at current speed.
        return StoppingResult(
            decel_mps2=0.0,
            coast_distance_m=distance_to_ped_m,
            braking_distance_m=0.0,
            stopping_distance_m=float("inf"),
            collision_avoided=False,
            impact_speed_mps=ego_speed_mps,
        )

    coast = ego_speed_mps * params.actuation_delay_s
    braking = (ego_speed_mps ** 2) / (2.0 * decel)
    stopping = coast + braking

    if stopping <= distance_to_ped_m:
        return StoppingResult(decel, coast, braking, stopping,
                              collision_avoided=True, impact_speed_mps=0.0)

    # Not enough room — compute speed at the pedestrian's position.
    dist_remaining_after_coast = max(0.0, distance_to_ped_m - coast)
    v_sq = ego_speed_mps ** 2 - 2.0 * decel * dist_remaining_after_coast
    impact = (v_sq ** 0.5) if v_sq > 0 else 0.0
    return StoppingResult(decel, coast, braking, stopping,
                          collision_avoided=False, impact_speed_mps=impact)

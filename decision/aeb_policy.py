"""
Staged AEB decision policy.

Escalation ladder (each stage requires the previous conditions plus a tighter TTC):
    NONE   -> nothing
    FCW    -> forward-collision warning (alert only)
    PARTIAL_AEB -> partial braking
    FULL_AEB    -> full braking

Braking only escalates when BOTH: TTC is below the stage threshold AND the model's
crossing probability is above the gate (so we don't hard-brake for a pedestrian the
model is confident is NOT crossing). Being inside the ego corridor is also required
for braking stages (a pedestrian safely on the sidewalk shouldn't trigger AEB).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class AEBLevel(IntEnum):
    NONE = 0
    FCW = 1
    PARTIAL_AEB = 2
    FULL_AEB = 3


LEVEL_LABEL = {
    AEBLevel.NONE: "CLEAR",
    AEBLevel.FCW: "WARNING",
    AEBLevel.PARTIAL_AEB: "BRAKE (partial)",
    AEBLevel.FULL_AEB: "BRAKE (full)",
}


@dataclass
class AEBThresholds:
    fcw_ttc_s: float = 2.6
    partial_aeb_ttc_s: float = 1.6
    full_aeb_ttc_s: float = 0.9
    crossing_prob_gate: float = 0.5

    @classmethod
    def from_config(cls, cfg: dict) -> "AEBThresholds":
        d = cfg.get("decision", {})
        return cls(
            fcw_ttc_s=d.get("fcw_ttc_s", 2.6),
            partial_aeb_ttc_s=d.get("partial_aeb_ttc_s", 1.6),
            full_aeb_ttc_s=d.get("full_aeb_ttc_s", 0.9),
            crossing_prob_gate=d.get("crossing_prob_gate", 0.5),
        )


@dataclass
class AEBDecision:
    level: AEBLevel
    ttc_s: float
    crossing_prob: float
    reason: str

    @property
    def is_braking(self) -> bool:
        return self.level >= AEBLevel.PARTIAL_AEB


def decide(
    ttc_s: float,
    crossing_prob: float,
    in_corridor: bool,
    thresholds: AEBThresholds,
) -> AEBDecision:
    """Pure function: map (TTC, crossing prob, corridor) -> escalation level."""
    gate = crossing_prob >= thresholds.crossing_prob_gate

    # Full / partial braking require the pedestrian to be in the path AND the
    # model to believe they're crossing.
    if in_corridor and gate and ttc_s <= thresholds.full_aeb_ttc_s:
        return AEBDecision(AEBLevel.FULL_AEB, ttc_s, crossing_prob,
                           "TTC critical + crossing likely + in path")
    if in_corridor and gate and ttc_s <= thresholds.partial_aeb_ttc_s:
        return AEBDecision(AEBLevel.PARTIAL_AEB, ttc_s, crossing_prob,
                           "TTC low + crossing likely + in path")
    # Warning is more permissive: warn even if slightly off-path or lower prob.
    if ttc_s <= thresholds.fcw_ttc_s and crossing_prob >= 0.35:
        return AEBDecision(AEBLevel.FCW, ttc_s, crossing_prob,
                           "pedestrian approaching path")
    return AEBDecision(AEBLevel.NONE, ttc_s, crossing_prob, "no conflict")

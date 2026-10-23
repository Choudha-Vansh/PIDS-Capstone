"""
Hazard detection for the Duckiebot live loop (detection-first).

A pedestrian is a "hazard" this frame when ALL hold:
    - detected as a person (comes from YOLO-pose via the tracker),
    - inside the ego travel corridor (small lateral offset from the bot's heading),
    - close enough to matter (apparent size / monocular distance gate).

This deliberately does NOT require the trained intent model: person-in-path is the
robust trigger for a first hardware run. The intent model is an optional gate you can
switch on later (hazard.use_intent_model: true) without touching the rest of the loop —
a clean place to "add intent" once the detection loop works.

Reuses the same geometry helpers as the simulation path so behaviour is consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from common.geometry import monocular_distance_from_height, lateral_offset_to_path
from decision.ttc import is_in_conflict


@dataclass
class HazardConfig:
    corridor_margin_m: float = 0.30
    min_bbox_height_frac: float = 0.25
    trigger_distance_m: float = 2.0
    use_intent_model: bool = False
    intent_prob_gate: float = 0.5
    focal_px: float = 1000.0
    assumed_ped_height_m: float = 1.7

    @classmethod
    def from_config(cls, cfg: dict) -> "HazardConfig":
        h = cfg.get("duckiebot", {}).get("hazard", {})
        m = cfg.get("monocular", {})
        return cls(
            corridor_margin_m=h.get("corridor_margin_m", 0.30),
            min_bbox_height_frac=h.get("min_bbox_height_frac", 0.25),
            trigger_distance_m=h.get("trigger_distance_m", 2.0),
            use_intent_model=h.get("use_intent_model", False),
            intent_prob_gate=h.get("intent_prob_gate", 0.5),
            focal_px=m.get("focal_px", 1000.0),
            assumed_ped_height_m=m.get("assumed_ped_height_m", 1.7),
        )


@dataclass
class HazardResult:
    hazard: bool
    track_id: int = -1
    distance_m: float = float("inf")
    lateral_m: float = float("inf")
    crossing_prob: Optional[float] = None
    reason: str = "clear"


def evaluate(dets, img_w: int, img_h: int, cfg: HazardConfig,
             intent_probs: Optional[dict] = None) -> HazardResult:
    """
    dets: list of perception.detector.Detection (with track_id, bbox_xyxy, center_px,
          bbox_height_px). intent_probs: optional {track_id: crossing_prob} if the
          intent model is enabled.
    Returns the single worst (closest in-path) hazard, or a clear result.
    """
    worst: Optional[HazardResult] = None
    for det in dets:
        if getattr(det, "track_id", -1) < 0:
            continue
        # too small in frame -> too far away to act on
        if det.bbox_height_px < cfg.min_bbox_height_frac * img_h:
            continue

        dist = monocular_distance_from_height(
            det.bbox_height_px, cfg.focal_px, cfg.assumed_ped_height_m)
        if dist > cfg.trigger_distance_m:
            continue

        cx, _ = det.center_px
        x_lat = (cx - img_w / 2.0) / cfg.focal_px * dist
        lateral = lateral_offset_to_path(x_lat)
        if not is_in_conflict(lateral, cfg.corridor_margin_m):
            continue

        prob = None
        if cfg.use_intent_model:
            prob = (intent_probs or {}).get(det.track_id)
            # only treat as hazard if the model also thinks they're crossing
            if prob is None or prob < cfg.intent_prob_gate:
                continue

        cand = HazardResult(True, det.track_id, dist, lateral, prob,
                            reason="person in path" if not cfg.use_intent_model
                            else "crossing pedestrian in path")
        if worst is None or cand.distance_m < worst.distance_m:
            worst = cand

    return worst if worst is not None else HazardResult(False)

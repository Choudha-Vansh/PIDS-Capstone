"""
End-to-end per-frame pipeline that ties perception -> 3D -> intent -> decision
together and produces a PipelineResult the dashboard renders. Works for the camera
(demo-video) path today; the same object plugs into the fusion path by swapping the
localization source (see `localize`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from common.geometry import (closing_speed_mps, lateral_offset_to_path)
from common.latency import LatencyBudget, ReactionTimer
from decision.aeb_policy import AEBDecision, AEBLevel, AEBThresholds, decide
from decision.ttc import time_to_collision, is_in_conflict
from decision.vehicle_dynamics import VehicleParams, simulate_stop
from intent.features import FeatureConfig, build_frame_feature, normalize_keypoints
from perception.kalman import KalmanCV3D
from perception.monocular_depth import localize_monocular


@dataclass
class TrackView:
    track_id: int
    bbox_xyxy: np.ndarray
    keypoints: np.ndarray
    distance_m: float
    crossing_prob: float
    decision: AEBDecision
    center_px: tuple
    in_path: bool               # True if inside the ego travel corridor (lane)


@dataclass
class PipelineResult:
    frame_idx: int
    tracks: List[TrackView]
    worst: Optional[TrackView]          # highest-severity track this frame
    ego_speed_mps: float
    latency: LatencyBudget
    reaction_ms: Optional[float]
    stop_avoided: Optional[bool]


class Pipeline:
    def __init__(self, cfg: dict, tracker, intent, fps: float):
        self.cfg = cfg
        self.tracker = tracker           # PoseTracker
        self.intent = intent             # OnlineIntent
        self.fps = fps
        self.dt = 1.0 / max(fps, 1e-6)
        self.fc = FeatureConfig(use_pose=cfg["intent"]["use_pose"],
                                use_3d=cfg["intent"]["use_3d"])
        self.thresholds = AEBThresholds.from_config(cfg)
        self.vparams = VehicleParams.from_config(cfg)
        self.mono = cfg["monocular"]
        self.reaction = ReactionTimer(fps=fps)
        self.kalmans: Dict[int, KalmanCV3D] = {}
        self.prev_dist: Dict[int, float] = {}
        self.last_seen: Dict[int, int] = {}
        self.max_age = cfg["perception"].get("track_buffer", 90)

    def localize(self, det, frame) -> float:
        """Return longitudinal distance (m). Camera path uses monocular height."""
        img_w = frame.image.shape[1]
        pos = localize_monocular(det, img_w, self.mono["focal_px"],
                                 self.mono["assumed_ped_height_m"])
        kf = self.kalmans.setdefault(det.track_id, KalmanCV3D(dt=self.dt))
        kf.update(pos)
        return float(kf.position[2])

    def process(self, frame) -> PipelineResult:
        lat = LatencyBudget()
        with lat.stopwatch("detect_track"):
            dets = self.tracker.track(frame.image)

        tracks: List[TrackView] = []
        worst: Optional[TrackView] = None
        reaction_ms = None
        stop_avoided = None
        img_h, img_w = frame.image.shape[:2]

        for det in dets:
            if det.track_id < 0:
                continue
            self.last_seen[det.track_id] = frame.frame_idx

            with lat.stopwatch("localize"):
                dist = self.localize(det, frame)

            # lateral position from monocular estimate
            cx, _ = det.center_px
            x_lat = (cx - img_w / 2.0) / self.mono["focal_px"] * dist
            lateral = lateral_offset_to_path(x_lat)

            # closing speed: prefer ego speed, else distance derivative
            prev = self.prev_dist.get(det.track_id, dist)
            self.prev_dist[det.track_id] = dist
            if frame.ego_speed_mps > 0.1:
                closing = frame.ego_speed_mps
            else:
                closing = closing_speed_mps(prev, dist, self.dt)
            ttc = time_to_collision(dist, closing)

            # intent
            with lat.stopwatch("intent"):
                bbox_norm = np.array([cx / img_w,
                                      (det.bbox_xyxy[1] + det.bbox_xyxy[3]) / 2 / img_h,
                                      (det.bbox_xyxy[2] - det.bbox_xyxy[0]) / img_w,
                                      det.bbox_height_px / img_h], np.float32)
                kp_norm = (normalize_keypoints(det.keypoints, img_w, img_h)
                           if self.fc.use_pose else None)
                feat = build_frame_feature(
                    bbox_norm, kp_norm,
                    np.array([x_lat, 0.0, dist]) if self.fc.use_3d else None,
                    None, lateral, frame.ego_speed_mps, self.fc)
                prob = self.intent.update(det.track_id, feat)
            prob = 0.0 if prob is None else prob

            in_path = is_in_conflict(lateral)
            with lat.stopwatch("decision"):
                dec = decide(ttc, prob, in_path, self.thresholds)

            tv = TrackView(det.track_id, det.bbox_xyxy, det.keypoints, dist, prob,
                           dec, det.center_px, in_path)
            tracks.append(tv)
            if worst is None or dec.level > worst.decision.level:
                worst = tv

        # reaction-time bookkeeping on the worst track
        if worst is not None:
            if worst.decision.level >= AEBLevel.FCW:
                self.reaction.note_hazard(frame.frame_idx)
            if worst.decision.is_braking:
                reaction_ms = self.reaction.note_brake(frame.frame_idx, lat.total_ms)
                res = simulate_stop(max(frame.ego_speed_mps, 1e-3),
                                    worst.distance_m, worst.decision.level,
                                    self.vparams)
                stop_avoided = res.collision_avoided
        else:
            self.reaction.clear()

        self._evict(frame.frame_idx)
        return PipelineResult(frame.frame_idx, tracks, worst,
                              frame.ego_speed_mps, lat, reaction_ms, stop_avoided)

    def _evict(self, frame_idx: int):
        stale = [t for t, s in self.last_seen.items()
                 if frame_idx - s > self.max_age]
        for t in stale:
            self.kalmans.pop(t, None); self.prev_dist.pop(t, None)
            self.last_seen.pop(t, None); self.intent.drop(t)

"""
Live Duckiebot loop: WiFi camera feed -> point-to-point pedestrian perception on the
laptop -> cruise / brake / resume decision -> drive commands back over the Duckietown
ROS stack.

    python run_car.py --veh <duckiebot-hostname>            # ROS camera + ROS drive
    python run_car.py --source stream --url http://<ip>:8080/stream --veh <host>
    python run_car.py --dry-run                             # NoOp actuator (no motion)
    python run_car.py --show                                # live overlay window

Behaviour: the bot cruises forward; when a pedestrian is detected IN ITS PATH and close
enough, it brakes to a stop; when the path is clear again it accelerates/resumes. Start
with --dry-run (default from config) and watch the printed decisions + overlay BEFORE
letting it move. See HARDWARE_DUCKIEBOT.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="Duckiebot pedestrian-AEB live loop")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--veh", help="Duckiebot hostname (overrides config.duckiebot.vehicle_name)")
    ap.add_argument("--source", choices=["ros", "stream"], help="camera ingest mode")
    ap.add_argument("--url", help="MJPEG/RTSP URL for --source stream")
    ap.add_argument("--dry-run", action="store_true",
                    help="force NoOp actuator (no motion), overriding config")
    ap.add_argument("--drive", action="store_true",
                    help="force real Duckiebot actuator (motion!), overriding config")
    ap.add_argument("--show", action="store_true", help="show a live overlay window")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = cfg.setdefault("duckiebot", {})
    if args.veh:
        d["vehicle_name"] = args.veh
    if args.source:
        d.setdefault("camera", {})["source"] = args.source
    if args.url:
        d.setdefault("camera", {})["stream_url"] = args.url
        d.setdefault("camera", {})["source"] = "stream"
    if args.dry_run:
        d.setdefault("safety", {})["dry_run"] = True
    if args.drive:
        d.setdefault("safety", {})["dry_run"] = False

    from sensors.duckiebot_source import DuckiebotCameraSource
    from perception.tracker import PoseTracker
    from control.hazard import HazardConfig, evaluate
    from control.cruise_policy import CruisePolicy, CruiseConfig
    from control.safety_stop import SafetyConfig, apply_safety_stop
    from sensors.tof_range import ToFRangeReader
    from hardware.actuator import build_actuator
    from common.latency import ReactionTimer

    source = DuckiebotCameraSource(cfg)
    tracker = PoseTracker(cfg["perception"]["pose_model"],
                          conf=cfg["perception"]["det_conf"],
                          tracker_cfg=cfg["perception"]["tracker"])
    hz_cfg = HazardConfig.from_config(cfg)
    policy = CruisePolicy(CruiseConfig.from_config(cfg))
    actuator = build_actuator(cfg)
    reaction = ReactionTimer(fps=source.fps)

    # independent ToF range safety net (fallback stop below a hard distance)
    safety_cfg = SafetyConfig.from_config(cfg)
    tof = ToFRangeReader(cfg)

    # optional intent model (detection-first by default; enable via hazard.use_intent_model)
    intent = None
    if hz_cfg.use_intent_model:
        from intent.online import OnlineIntent
        from intent.features import FeatureConfig, build_frame_feature, normalize_keypoints
        intent = OnlineIntent(cfg)
        fc = FeatureConfig(use_pose=cfg["intent"]["use_pose"], use_3d=cfg["intent"]["use_3d"])

    dry = cfg["duckiebot"]["safety"].get("dry_run", True)
    print(f"Duckiebot '{cfg['duckiebot']['vehicle_name']}'  mode={cfg['duckiebot']['camera']['source']}"
          f"  actuator={'NoOp (dry-run)' if dry else 'DUCKIEBOT (LIVE MOTION)'}")
    if not dry:
        print("!! LIVE MOTION ENABLED — keep the e-stop/shutdown button within reach !!")

    try:
        for frame in source:
            img_h, img_w = frame.image.shape[:2]
            dets = tracker.track(frame.image)

            intent_probs = None
            if intent is not None:
                intent_probs = {}
                for det in dets:
                    if det.track_id < 0:
                        continue
                    cx, _ = det.center_px
                    bbox_norm = _bbox_norm(det, img_w, img_h)
                    kp = normalize_keypoints(det.keypoints, img_w, img_h) if fc.use_pose else None
                    feat = build_frame_feature(bbox_norm, kp, None, None, 0.0,
                                               frame.ego_speed_mps, fc)
                    p = intent.update(det.track_id, feat)
                    if p is not None:
                        intent_probs[det.track_id] = p

            # LAYER 1 (smart): camera pedestrian hazard -> cruise/brake/resume
            hzr = evaluate(dets, img_w, img_h, hz_cfg, intent_probs)
            cam_cmd = policy.step(hzr.hazard)

            # LAYER 2 (safety net): independent ToF range -> force stop under a hard
            # distance, even if the camera layer didn't brake. Can only slow/stop.
            tof_m = tof.read()
            frame.tof_distance_m = tof_m
            cmd = apply_safety_stop(cam_cmd, tof_m, safety_cfg)
            tof_stop = cmd is not cam_cmd

            # reaction-time bookkeeping: hazard onset -> brake command (either layer)
            if hzr.hazard or tof_stop:
                reaction.note_hazard(frame.frame_idx)
            rt = reaction.note_brake(frame.frame_idx) if cmd.braking else None

            actuator.drive(cmd.v, cmd.omega)
            source.set_ego_speed(cmd.v)

            if tof_stop:
                tag = "TOF-STOP"
            elif cmd.braking:
                tag = "BRAKE"
            elif cmd.v > 0:
                tag = "CRUISE"
            else:
                tag = "STOP"
            extra = f" ped={hzr.distance_m:.2f}m id={hzr.track_id}" if hzr.hazard else ""
            ttxt = f" tof={tof_m*100:.0f}cm" if tof_m is not None else " tof=--"
            rtxt = f"  reaction={rt:.0f}ms" if rt else ""
            print(f"[{frame.frame_idx:05d}] {tag:8s} v={cmd.v:.2f}  {cmd.reason}{extra}{ttxt}{rtxt}")

            if args.show:
                if not _show(frame.image, dets, hzr, cmd, tof_m):
                    break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        actuator.stop()          # always leave the bot stopped
        actuator.close()
        source.close()
        print(f"Stopped. Reaction summary: {reaction.summary()}")


def _bbox_norm(det, img_w, img_h):
    import numpy as np
    cx, _ = det.center_px
    return np.array([cx / img_w,
                     (det.bbox_xyxy[1] + det.bbox_xyxy[3]) / 2 / img_h,
                     (det.bbox_xyxy[2] - det.bbox_xyxy[0]) / img_w,
                     det.bbox_height_px / img_h], dtype="float32")


def _show(img, dets, hzr, cmd, tof_m=None):
    import cv2
    vis = img.copy()
    for det in dets:
        if det.track_id < 0:
            continue
        x0, y0, x1, y1 = [int(v) for v in det.bbox_xyxy]
        is_hz = hzr.hazard and det.track_id == hzr.track_id
        color = (0, 0, 255) if is_hz else (0, 200, 0)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
    tof_stop = "ToF SAFETY STOP" in cmd.reason
    banner = ("TOF-STOP" if tof_stop else
              "BRAKE" if cmd.braking else ("CRUISE" if cmd.v > 0 else "STOP"))
    bcol = (0, 0, 255) if cmd.braking else (0, 180, 0)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), bcol, -1)
    cv2.putText(vis, f"{banner}  v={cmd.v:.2f}  {cmd.reason}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    tof_txt = f"ToF: {tof_m*100:.0f} cm" if tof_m is not None else "ToF: --"
    cv2.putText(vis, tof_txt, (10, vis.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Duckiebot ADAS", vis)
    return cv2.waitKey(1) & 0xFF != ord("q")


if __name__ == "__main__":
    main()

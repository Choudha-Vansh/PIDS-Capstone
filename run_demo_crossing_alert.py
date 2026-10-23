"""
Multi-pedestrian crossing-alert demo view.

Unlike run_demo.py (which shows every tracked pedestrian with the full TTC/prob
panel), this draws a simpler box + label per pedestrian -- "Crossing" while they're
inside the car's driving lane/path, "Not Crossing" otherwise -- plus a black alert
strip below the video that reads "Brake Engaged" whenever ANY tracked pedestrian is
currently in-lane.

Every detection the tracker reports as a person is shown; there is no single-target
lock, so a stray misdetection just shows up as its own box rather than silently
hijacking the only tracked slot.

    python run_demo_crossing_alert.py --video path/to/demo.mp4 [--ego-speed 8.3]
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PANEL_HEIGHT_PX = 100
IN_LANE_COLOR = (0, 0, 255)      # red  (BGR) -- Crossing
OUT_LANE_COLOR = (0, 200, 0)     # green -- Not Crossing


def _draw_track(img, tv, label, color):
    x0, y0, x1, y1 = [int(v) for v in tv.bbox_xyxy]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (x0, y0 - th - 10), (x0 + tw + 8, y0), color, -1)
    cv2.putText(img, label, (x0 + 4, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)


def _compose_frame(img, brake_engaged: bool):
    h, w = img.shape[:2]
    panel = np.zeros((PANEL_HEIGHT_PX, w, 3), dtype=np.uint8)
    if brake_engaged:
        text = "Brake Engaged"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(panel, text, ((w - tw) // 2, (PANEL_HEIGHT_PX + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return np.vstack([img, panel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to demo video")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--ego-speed", type=float, default=0.0,
                    help="Constant ego speed (m/s) to assume for TTC")
    ap.add_argument("--output", default=None,
                    help="Output mp4 path (default results/crossing_alert_demo.mp4)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    from sensors.video_source import VideoSource
    from perception.tracker import PoseTracker
    from intent.online import OnlineIntent
    from pipeline import Pipeline

    source = VideoSource(args.video, ego_speed_mps=args.ego_speed)
    if abs(source.fps - 30) > 3:
        print(f"WARNING: video is {source.fps:.1f} fps; velocity/TTC features assume "
              f"~30 fps. Consider resampling.")

    tracker = PoseTracker(cfg["perception"]["pose_model"],
                          conf=cfg["perception"]["det_conf"],
                          tracker_cfg=cfg["perception"]["tracker"])
    intent = OnlineIntent(cfg)
    pipe = Pipeline(cfg, tracker, intent, fps=source.fps)

    out_path = args.output or os.path.join(cfg["paths"]["results"],
                                           "crossing_alert_demo.mp4")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    out = None
    n_crossing_frames = 0
    n_frames = 0

    for frame in source:
        r = pipe.process(frame)
        img = frame.image.copy()

        brake_engaged = False
        for tv in r.tracks:
            label = "Crossing" if tv.in_path else "Not Crossing"
            color = IN_LANE_COLOR if tv.in_path else OUT_LANE_COLOR
            _draw_track(img, tv, label, color)
            brake_engaged = brake_engaged or tv.in_path

        out_frame = _compose_frame(img, brake_engaged)
        if out is None:
            h, w = out_frame.shape[:2]
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                  source.fps, (w, h))
        out.write(out_frame)
        n_frames += 1
        if brake_engaged:
            n_crossing_frames += 1
        if r.frame_idx % 100 == 0:
            print(f"  frame {r.frame_idx}  tracks={len(r.tracks)}")

    if out:
        out.release()
    print(f"Wrote {out_path}")
    print(f"Brake engaged on {n_crossing_frames}/{n_frames} frames")


if __name__ == "__main__":
    main()

"""
Run the full camera pipeline on a demo video and show the live dashboard:

    python run_demo.py --video path/to/demo.mp4 [--ego-speed 8.3] [--headless]

Camera path: YOLO-pose detect+track -> monocular 3D -> intent -> TTC -> AEB -> dashboard.
Requires a trained model at weights/intent_model.pth (see SETUP.md). --headless writes
an annotated output video + CSV log without opening a window (useful on a server).
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to demo video")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--ego-speed", type=float, default=0.0,
                    help="Constant ego speed (m/s) to assume for TTC (dashcam has none)")
    ap.add_argument("--headless", action="store_true",
                    help="No window; write annotated mp4 + CSV instead")
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

    if args.headless:
        _run_headless(source, pipe, cfg)
    else:
        from dashboard.app import run
        run(source, pipe, cfg)


def _run_headless(source, pipe, cfg):
    import cv2
    from dashboard.overlays import TrailStore, draw_track
    from dashboard.telemetry import TelemetryLogger
    trails = TrailStore(cfg["dashboard"].get("trail_length", 40))
    out = None
    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["results"], "demo_annotated.mp4")
    log_path = cfg["dashboard"].get("export_log", "results/session_log.csv")
    telemetry = TelemetryLogger(log_path)
    for frame in source:
        r = pipe.process(frame)
        img = frame.image.copy()
        for tv in r.tracks:
            draw_track(img, tv, trails)
        if out is None:
            h, w = img.shape[:2]
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                  source.fps, (w, h))
        out.write(img)
        telemetry.log(r)
        if r.frame_idx % 100 == 0:
            print(f"  frame {r.frame_idx}")
    if out:
        out.release()
    telemetry.close()
    print(f"Wrote {out_path} and {log_path}. Reaction summary: {pipe.reaction.summary()}")


if __name__ == "__main__":
    main()

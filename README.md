# ADAS Pedestrian Intent + AEB Simulator

Software-only (no hardware) pipeline that detects pedestrians, tracks them point-to-point
(17 body keypoints + fused 3D position), predicts **crossing / not-crossing** with a
calibrated temporal model, and drives a staged **Automatic Emergency Braking** decision
with a measured system reaction time — all shown live in a PyQt dashboard.

> **Current phase: SIMULATION ONLY.** Camera/radar/LIDAR come from datasets
> (nuScenes) and recorded video (PIE/JAAD) or your demo clip. The ego-vehicle and
> braking are simulated. Real sensors + Jetson are a future phase, gated on results.

## Two input paths (by design)
| Path | Sensors | Trained/validated on | What can test it |
|---|---|---|---|
| **Camera** | camera only | PIE + JAAD | your demo video (dashcam has only a camera) |
| **Fusion** | camera + radar + LIDAR | nuScenes | nuScenes scenes (demo video has no radar/LIDAR) |

Both feed the **same** decision/AEB layer through the `sensors/base.py` abstraction,
so the hardware swap later is localized.

## Pipeline
```
sensor frame ─► detect + pose (YOLO-pose) ─► track (BoT-SORT, stable IDs)
             ─► 3D localize (fusion on nuScenes / monocular on demo video) + Kalman
             ─► temporal features ─► intent model ─► calibrated crossing prob
             ─► TTC + staged AEB policy + vehicle dynamics ─► dashboard + log
```

## Layout
```
config/        central YAML config (no hardcoded paths)
common/        latency instrumentation, geometry / monocular distance
sensors/       SensorSource interface + video + nuScenes sources
perception/    detector, tracker, radar/LIDAR fusion + Kalman, monocular depth
intent/        feature builder, temporal model, train / evaluate / calibrate
data/          PIE + JAAD prep -> sequences, split by clip
decision/      TTC, staged AEB policy, vehicle-dynamics stopping model
eval/          metrics (mAP, MOTA/IDF1, P/R/F1/AUC/AP, ECE, reaction-time stats)
dashboard/     PyQt app (video + overlays + telemetry/warning panel)
runtime/       (deferred) ONNX/TensorRT export + Jetson benchmark
tests/         pure-python unit tests (TTC, dynamics, features, latency)
run_demo.py    end-to-end camera path on a demo video
```

See **SETUP.md** for install + dataset + run instructions.

## Quick start (after SETUP.md)
```bash
python data/prepare_jaad.py && python data/prepare_pie.py && python data/combine.py
python intent/train.py && python intent/calibrate.py && python intent/evaluate.py
python run_demo.py --video path/to/demo.mp4
```

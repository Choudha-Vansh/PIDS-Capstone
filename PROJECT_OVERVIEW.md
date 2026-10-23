# Project Overview — PIDS-Capstone (ADAS Pedestrian Intent + AEB Simulator)

*Paste this into a new chat to give context about the whole project.*

## What it is
A software-only ADAS pipeline that detects pedestrians, tracks them precisely
(17 pose keypoints + a fused 3D position/velocity), predicts **crossing vs
not-crossing** with a calibrated temporal model, and drives a staged **Automatic
Emergency Braking (AEB)** decision with a **measured system reaction time** — all
shown live in a PyQt desktop dashboard.

## Current scope (important)
**SIMULATION ONLY — no hardware yet.** Sensor data comes from datasets (nuScenes)
and recorded video (PIE/JAAD/a demo clip). The ego-vehicle motion and braking are
simulated. Moving to a real prototype car with camera+radar+LIDAR on a Jetson Nano
(fallback: Jetson Orin Nano) is a **future phase, gated on simulation sign-off.**
All sensor input is isolated behind a `SensorFrame`/`SensorSource` abstraction so the
hardware swap later is localized, not a rewrite. ONNX/TensorRT export exists but is
deferred.

## Pipeline (per frame)
```
sensor frame → detect + pose (YOLO-pose) → track (BoT-SORT, stable IDs)
             → 3D localize (radar/LIDAR fusion on nuScenes | monocular on video) + Kalman
             → temporal features → intent model → calibrated crossing probability
             → TTC + staged AEB (FCW→partial→full) + vehicle dynamics
             → PyQt dashboard + CSV telemetry log
```

## Two input paths (kept separate by design)
| Path | Sensors | Trained/validated on | Testable with |
|---|---|---|---|
| **Camera** | camera only | PIE + JAAD | a plain demo video / dashcam |
| **Fusion** | camera + radar + LIDAR | nuScenes | nuScenes scenes |
Both feed the **same** decision/AEB layer. A plain demo video can only exercise the
camera path (it has no radar/LIDAR) — that's physics, not a bug.

## Tech stack
Python, PyTorch (temporal intent model), Ultralytics YOLO-pose + BoT-SORT
(detection/pose/tracking), OpenCV (video + overlays), PyQt5 (dashboard), NumPy/SciPy,
scikit-learn (metrics), nuscenes-devkit (fusion path). Config-driven via a single
`config/config.yaml`.

## Repo layout (`C:\Users\vansh\Desktop\PIDS-Capstone`)
```
config/config.yaml   central config — ONLY place for paths/thresholds/model choices
common/              latency.py (LatencyBudget, ReactionTimer), geometry.py (monocular dist)
sensors/             base.py (SensorFrame/SensorSource), video_source.py, nuscenes_source.py
perception/          detector.py (YOLO-pose), tracker.py (BoT-SORT), fusion.py (radar/LIDAR),
                     kalman.py (CV 3D filter), monocular_depth.py (camera-only distance)
intent/              features.py (shared feature builder), model.py (GRU/Transformer),
                     dataset.py, train.py, calibrate.py (temp scaling), evaluate.py,
                     online.py (live inference)
data/                prepare_jaad.py, prepare_pie.py, combine.py (clip-disjoint splits)
decision/            ttc.py, aeb_policy.py (staged FCW→partial→full), vehicle_dynamics.py
eval/                metrics.py (P/R/F1/AUC/AP + ECE calibration)
dashboard/           overlays.py (OpenCV drawing), telemetry.py (CSV logger), app.py (PyQt)
runtime/             export_onnx.py (DEFERRED — future Jetson phase)
tests/               test_decision.py, test_kalman.py, test_features.py (pure-python)
pipeline.py          per-frame orchestration → PipelineResult
run_demo.py          entry point: camera pipeline on a demo video → dashboard (or --headless)
CLAUDE.md            agent context/conventions for Claude Code
SETUP.md             install + dataset/model download + run instructions
```

## Feature vector (intent model input)
Per pedestrian per frame, `intent/features.py` builds a **46-dim** vector when
`use_pose=true, use_3d=true`: 34 pose keypoints (17×xy) + 4 bbox + 3 position +
3 velocity + 1 distance-to-ego-path + 1 ego-speed. Sequences are 16 frames, predicting
crossing 8 frames ahead. The **same builder is used for training and live inference**
so distributions match.

## Key design rules (non-negotiable conventions)
- No hardcoded paths/magic numbers — everything in `config/config.yaml`.
- Train/inference feature parity — one shared feature builder.
- No zero-padded sequences at inference — predict only once a full real window exists.
- Gap-aware sequences — never differentiate velocity across dropped/occluded frames.
- Clean train/val/test split **by clip** (no frame leakage); metrics only on the
  untouched test split.
- Model outputs raw logits; sigmoid at inference; training uses
  `BCEWithLogitsLoss(pos_weight=...)` for class imbalance.
- Stable track IDs via BoT-SORT with a long `track_buffer`; stale per-track state
  evicted after `track_buffer` frames.

## Datasets & models
- **PIE** (primary intent; has ego-vehicle OBD speed for TTC) + **JAAD** (secondary).
- **nuScenes** (camera+radar+LIDAR, start with v1.0-mini) for the fusion path.
- **YOLO-pose** (yolo11n-pose / yolov8n-pose) auto-downloaded; BoT-SORT ships with
  Ultralytics. All dataset downloads are registration/EULA-gated and done manually.

## Metrics / acceptance criteria
Detection mAP + keypoint accuracy; tracking MOTA/IDF1 + ID-switch count; intent
precision/recall/F1/AUC/AP + calibration (ECE) on the test split; TTC error; AEB
engagement precision/recall + collision-avoided rate; per-stage + total latency and
reaction-time distribution (mean/p95).

## Status (as of this build)
- Pure-python core DONE and unit-tested: decision/TTC/vehicle-dynamics (11 tests),
  Kalman (2), feature builder + gap-aware slicing (6). All pass.
- Data pipeline verified end-to-end on JAAD: 686 tracks → ~116k sequences
  (16×46) → clip-disjoint train/val/test, no leakage.
- Torch/Ultralytics/PyQt code written and compiles; not yet run on GPU (needs the dev
  machine). `prepare_jaad.py` now runs YOLO-pose over JAAD clips to fill real keypoints
  (with resumable checkpointing). A `.venv` is set up in the project.
- **No trained weights yet** — `weights/intent_model.pth` is produced by
  `python intent/train.py`; it is not a download. YOLO-pose weights download at setup.

## How to run (on the dev machine; see SETUP.md)
```bash
pip install -r requirements.txt   # + CUDA-matched torch
python data/prepare_jaad.py && python data/prepare_pie.py && python data/combine.py
python intent/train.py && python intent/calibrate.py && python intent/evaluate.py
python run_demo.py --video path/to/demo.mp4 --ego-speed 8.3    # add --headless for no window
```

## Open TODOs / gotchas
- JAAD XML has no keypoints → filled via YOLO-pose in `prepare_jaad.py` (or set
  `use_pose:false`); decide before training.
- PIE OBD-speed file layout varies by release (`prepare_pie.py` warns if speed absent).
- Monocular `focal_px` is a guess — calibrate per camera or camera-path TTC is biased.
- nuScenes 3D-error eval vs ground-truth boxes is stubbed in `fusion.py` (real TODO).
- Changing feature flags changes the input dim → retrain from scratch.
- Jetson Nano 4GB real-time feasibility with pose+fusion+intent is the top hardware
  risk (Orin Nano is the fallback).
```

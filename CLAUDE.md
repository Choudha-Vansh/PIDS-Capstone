# CLAUDE.md — Project context for Claude Code

Read this first. It tells you what this project is, how it's laid out, the rules to
follow, and what state things are in. Keep it up to date as the project evolves.

## What this is

An ADAS pedestrian-safety pipeline:

    sensor frame → detect + pose (YOLO-pose) → track (BoT-SORT, stable IDs)
                 → 3D localize (radar/LIDAR fusion on nuScenes / monocular on video) + Kalman
                 → temporal features → intent model → calibrated crossing probability
                 → TTC + staged AEB policy + vehicle dynamics → PyQt dashboard + CSV log

The goal is precise per-pedestrian tracking (17 pose keypoints + fused 3D position),
a calibrated crossing / not-crossing prediction, a staged Automatic Emergency Braking
decision, and a **measured** system reaction time — all shown live in a desktop dashboard.

## CURRENT SCOPE — SIMULATION ONLY (no hardware)

Everything runs in software in VS Code. **No physical sensors, no embedded hardware
in this phase.** Camera/radar/LIDAR come from datasets (nuScenes) and recorded video
(PIE/JAAD/your demo clip). The ego-vehicle and braking are simulated.

The move to a prototype car with real sensors on a Jetson controller is a **future
phase, out of scope until simulation results are signed off.** Do NOT add hardware
drivers, sensor SDKs, or device-specific code now. DO keep the architecture
hardware-portable: all sensor input is isolated behind `sensors/base.py`, so swapping
simulated feeds for real ones later is a localized change, not a rewrite.

Future hardware target (deferred): Jetson Nano 4GB, fallback Jetson Orin Nano if the
Nano can't hit real-time. ONNX/TensorRT export lives in `runtime/export_onnx.py` and
is intentionally NOT part of the current deliverable.

## Two input paths (by design — don't conflate them)

| Path | Sensors | Trained/validated on | What can test it |
|---|---|---|---|
| **Camera** | camera only | PIE + JAAD | a plain demo video / dashcam (only has a camera) |
| **Fusion** | camera + radar + LIDAR | nuScenes | nuScenes scenes (a demo video has no radar/LIDAR) |

Both feed the **same** decision/AEB layer through the `sensors.SensorFrame` abstraction.
A demo video can only exercise the camera path; that is a physical fact about the inputs,
not a limitation to "fix".

## Repo layout

```
config/config.yaml     central config — the ONLY place for paths/thresholds/model choices
common/                latency instrumentation (LatencyBudget, ReactionTimer), geometry
sensors/               SensorSource ABC + SensorFrame; VideoSource, NuScenesSource
perception/            detector (YOLO-pose), tracker (BoT-SORT), fusion (radar/LIDAR),
                       kalman (CV 3D filter), monocular_depth (camera-only distance)
intent/                features (shared builder), model (GRU/Transformer), dataset,
                       train, calibrate (temperature scaling), evaluate, online (live inference)
data/                  prepare_jaad, prepare_pie, combine (clip-disjoint train/val/test)
decision/              ttc, aeb_policy (staged FCW→partial→full), vehicle_dynamics
eval/                  metrics (P/R/F1/AUC/AP + ECE)
dashboard/             overlays (OpenCV drawing), app (PyQt window + CSV log)
runtime/               export_onnx (DEFERRED — future Jetson phase)
tests/                 pure-python unit tests (no torch/GPU needed)
pipeline.py            per-frame orchestration → PipelineResult
run_demo.py            entry point: camera pipeline on a demo video → dashboard
```

## Non-negotiable conventions

- **No hardcoded paths or magic numbers in code.** Everything configurable lives in
  `config/config.yaml`; code reads it. This was a real bug in the earlier prototype.
- **Train / inference feature parity.** `intent/features.py` builds the feature vector
  for BOTH offline dataset prep and live inference. If you change the feature layout,
  change it there once — never build features differently in the two paths.
- **No zero-padded sequences at inference.** `intent/online.py` only predicts once a
  FULL real window exists, matching how training windows are built. Don't reintroduce
  warmup/zero-padding — it caused a train/inference distribution mismatch before.
- **Gap-aware sequences.** `slice_sequences` skips windows that span a frame gap
  (dropped/occluded frames); velocity is never differentiated across a gap. Preserve this.
- **Clean train/val/test separation, split BY CLIP.** `data/combine.py` splits on unique
  clip IDs so frames from one video never span splits. Report metrics only on the
  untouched **test** split (val is for model selection + calibration).
- **Model outputs raw logits.** Sigmoid is applied at inference/eval, and training uses
  `BCEWithLogitsLoss(pos_weight=...)` for class imbalance. Don't add a sigmoid into the
  model's forward().
- **Stable track IDs matter.** Tracking uses BoT-SORT with a long `track_buffer` so brief
  occlusions don't spawn new IDs; per-track state is evicted after `track_buffer` frames.

## Build / run commands

```bash
# setup (see SETUP.md for CUDA-matched torch + dataset downloads)
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# data → train → calibrate → evaluate
python data/prepare_jaad.py && python data/prepare_pie.py && python data/combine.py
python intent/train.py && python intent/calibrate.py && python intent/evaluate.py

# live dashboard on a demo video (camera path)
python run_demo.py --video path/to/demo.mp4 --ego-speed 8.3
# headless annotated render instead of a window:
python run_demo.py --video path/to/demo.mp4 --headless

# fusion smoke test on nuScenes-mini
python perception/fusion.py --scene 0

# tests (pure python, no GPU)
python tests/test_decision.py && python tests/test_kalman.py && python tests/test_features.py
```

## Current state (as of initial build)

- Pure-python core is DONE and unit-tested: decision/TTC/vehicle-dynamics (11 tests),
  Kalman (2), feature builder + gap-aware slicing (6). All pass.
- Data pipeline verified end-to-end on JAAD annotations: 686 tracks → ~116k sequences
  (16 frames × 46 features) → clip-disjoint train/val/test.
- Torch / ultralytics / PyQt code is written and compiles, but was NOT run in the build
  environment (no GPU there). First real run happens on the dev machine.

## Known TODOs / gotchas

- **JAAD has no pose keypoints in its XML.** `prepare_jaad.py` leaves keypoint slots
  zero. To train pose features on JAAD, either run YOLO-pose over the JAAD clips to fill
  them, or set `intent.use_pose: false` for the dataset-trained model and let pose come
  from the live path. Decide this explicitly before training with `use_pose: true`.
- **PIE annotation layout varies by release.** `prepare_pie.py` is defensive about the
  OBD-speed file location; verify ego-speed is actually populated (it warns if not).
- **Monocular distance needs calibration.** `config.monocular.focal_px` is a guess;
  calibrate per camera or the camera-path TTC will be biased.
- **nuScenes fusion 3D-error eval vs ground-truth boxes is stubbed** in `fusion.py`
  (`_demo` just confirms frames load). Wiring up the nuScenes eval API is a real TODO.
- **Feature dim is 46** with `use_pose=true, use_3d=true` (34 pose + 4 bbox + 3 pos +
  3 vel + 1 dist + 1 ego). Changing feature flags changes the dim; retrain from scratch.

## Roadmap / planned modules

- **Traffic-light perception + signal-aware braking + interactive dashboard** (next
  planned module; spec in `ADAS_TRAFFIC_LIGHT_PROMPT.md`). **Camera-only** (radar/LIDAR
  cannot sense color or digits). Adds `perception/traffic_light.py` (detect + classify
  RED/YELLOW/GREEN, distance, optional best-effort close-range countdown digit
  recognition) and `decision/signal_policy.py` (a pure, unit-tested state machine).
  Key policy: keep **signal compliance** (red=STOP, yellow=SLOW, green=GO — a HUD
  Slow/Stop/Go driving-intent indicator) SEPARATE from **pedestrian AEB**. Do NOT raise
  an emergency brake merely because the light is red; the primary pedestrian-brake
  trigger is GREEN + a pedestrian in-path and moving (overrides GO), with an
  imminent-collision safety net in all states. Slow/Stop/Go comes from color state
  (robust from afar); the numeric countdown timer is optional/close-range only and must
  fall back to state-only. Dashboard gains transport + overlay controls, per-track
  detail, and live threshold sliders. Telemetry columns are APPENDED (don't reorder).
- **Duckiebot hardware bring-up** — STARTED (camera path). The live loop is built and
  spec'd in `HARDWARE_DUCKIEBOT.md`: the Duckiebot's 5 ft webcam streams over WiFi to the
  laptop, which runs YOLO-pose point-to-point perception and sends cruise/brake/resume
  commands back over the Duckietown ROS stack. New modules (all behind the existing
  abstractions): `sensors/duckiebot_source.py` (WiFi camera → SensorFrame, ROS or stream),
  `hardware/actuator.py` (`ActuatorInterface` + `NoOpActuator` dry-run + `DuckiebotROSActuator`),
  `control/cruise_policy.py` (cruise/brake/resume state machine, unit-tested),
  `control/hazard.py` (detection-first person-in-path gate, intent-model hook), `run_car.py`
  (entry point). Config lives under `duckiebot:` in `config.yaml`. Detection-first by
  default; the trained intent model is an opt-in gate (`duckiebot.hazard.use_intent_model`).
  Starts in dry-run (NoOp actuator) — enable motion only after decisions look right.
  TWO-LAYER BRAKING: layer 1 is the camera pedestrian decision (above); layer 2 is an
  INDEPENDENT ToF range safety net (`sensors/tof_range.py` + `control/safety_stop.py`,
  unit-tested) that forces a stop when any object is within
  `duckiebot.tof.safety_stop_distance_m` (default 30 cm), regardless of the camera — it
  can only slow/stop, never speed up. Catches missed/late camera braking and
  non-pedestrian obstacles; the ToF is a low-mounted fallback, not a pedestrian-aware signal.
  Note: this moves beyond the earlier "simulation only" scope above; hardware code is
  isolated and kept hardware-portable per the architecture rules, which the user signed
  off on by starting this phase.

## Working style in this repo

Act like an ADAS engineer, not a code generator. State assumptions, flag risks, and at
each phase gate propose the approach and wait for confirmation before writing large
amounts of code. When a real design choice comes up (model variant, thresholds, fusion
strategy), present options with trade-offs rather than silently picking one.

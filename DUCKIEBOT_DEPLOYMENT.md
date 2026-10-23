# Duckiebot Deployment Plan — Pedestrian-Intent ADAS (camera path)

Goal: run the existing camera-path pipeline (pose → track → intent → TTC → staged
AEB) on a **Duckietown Duckiebot** (Jetson Nano 4GB, 160° front camera, wheel encoders,
front ToF, IMU, 2 DC motors, 4 RGB LEDs). Traffic-light/timer is out of scope for now.

> Assumed defaults for first bring-up (change any of these and the plan adapts):
> **compute = offboard laptop first**, **integration = Duckietown ROS node**,
> **brake action = warn-first, then enable motor-stop**. Pedestrian-target
> recommendation is in §1 — read that first, it's the make-or-break decision.

---

## 0. Why the Duckiebot is a good fit

The Duckiebot natively provides the two signals the simulation had to fake:

| Pipeline need | Simulation source | Duckiebot source |
|---|---|---|
| Ego speed (`SensorFrame.ego_speed_mps`) | assumed constant / OBD | **wheel encoders** (hall effect) |
| Distance to pedestrian (`distance_m`) | monocular height estimate | **front ToF sensor** (metric, direct) |
| Camera frame | video file | **160° front camera** (ROS topic) |
| "Brake" actuation | simulated stop | **DC motors** (stop) + **RGB LEDs** (indicator) |

So the `sensors/base.py` abstraction pays off here: we add a Duckiebot source + an
actuator, and the perception/intent/decision code is unchanged.

## 1. THE KEY DECISION — what plays the "pedestrian" (read this first)

Your intent model is trained on **JAAD = real adult humans** with 17-keypoint body
pose. Two hard facts about the Duckiebot camera: it sits ~10 cm off the floor with a
160° fisheye, so (a) a standing adult is seen from below as mostly legs, and (b)
Duckietown "pedestrians" are usually **rubber ducks**, which YOLO-pose will not detect
as people at all. Either way the JAAD-trained pose model does **not** transfer as-is.

**Recommendation (most feasible, least rework):** for the Duckiebot phase, treat the
pedestrian as a **small human figure / doll at Duckiebot scale placed at the roadside**,
and switch the intent model to a **pose-free (bbox + kinematics) variant** —
`intent.use_pose: false`. Rationale:
- The fisheye-from-below view makes reliable 17-keypoint pose unrealistic; bbox +
  motion (approach speed, lateral movement toward the lane) is robust at this scale.
- Your feature builder already supports `use_pose:false` (feature dim drops from 46 to
  12: 4 bbox + 3 pos + 3 vel + 1 dist + 1 ego). Retrain PIE/JAAD with `use_pose:false`
  and you get a model whose inputs the Duckiebot can actually produce.
- "Crossing" becomes: a figure moving laterally into the bot's travel corridor — which
  the bbox-centroid trajectory + ToF distance capture well.

Alternative if you insist on real humans: mount the camera higher, or accept detection
only when a person is several meters ahead (upper body in frame). Alternative if you
want ducks: retrain YOLO to detect ducks and redefine "intent" as duck-in-path motion
(no skeleton) — more work, drops pose entirely.

**Action:** decide this now. The rest of the plan assumes the pose-free figure approach.

## 2. Software architecture (Duckietown is ROS-based)

The Duckietown stack runs ROS inside Docker, orchestrated by the Duckietown Shell
(`dts`). The camera and motors are ROS topics. We add one custom ROS node.

```
camera_node/image/compressed ─► [adas_node] ─► car_cmd (Twist2DStamped)
   wheel encoders ─────────────►             └► LED command (brake indicator)
   front ToF ──────────────────►             └► /adas/telemetry (for dashboard)
```

`adas_node` responsibilities:
1. Subscribe to the camera topic; decompress to a BGR image.
2. Read wheel-encoder ticks → ego speed; read ToF → distance ahead.
3. Build a `SensorFrame` (image + ego_speed_mps + a ToF-backed distance) and call the
   existing `Pipeline.process()`.
4. Map the `AEBDecision` to a motor command (slow / stop) and an LED color, and publish
   telemetry so the dashboard can render remotely.

> Confirm the exact topic names on YOUR Duckiebot image (they're namespaced by the
> robot hostname, e.g. `/<bot>/camera_node/image/compressed`,
> `/<bot>/left_wheel_encoder_node/tick`, `/<bot>/front_center_tof_driver_node/range`,
> `/<bot>/car_cmd_switch_node/cmd`, `/<bot>/led_emitter_node/...`). They vary a little by
> Duckietown software version — list them with `rostopic list` before wiring the node.

## 3. New code to add (kept behind existing abstractions)

- `sensors/duckiebot_source.py` — a `SensorSource` that yields `SensorFrame`s from ROS
  topics (camera + encoder-derived speed + ToF distance). This is the ROS-facing seam;
  everything downstream stays pure Python.
- `hardware/actuator.py` — an `ActuatorInterface` with two implementations: a **no-op /
  sim** actuator (safe, logs only) and a **Duckiebot** actuator (publishes car_cmd +
  LED). The decision layer already produces the brake level; this just executes it.
- `perception/monocular_depth.py` stays as a fallback, but distance now prefers the ToF
  reading when available (add `tof_distance_m` to `SensorFrame`).
- `run_duckiebot.py` (or the ROS node entrypoint) — wires `DuckiebotSource` + `Pipeline`
  + `ActuatorInterface`, mirroring `run_demo.py`.
- Config block `duckiebot:` in `config/config.yaml` — topic names, wheel radius +
  encoder ticks/rev (for speed), ToF max range, motor stop/slow gains, LED colors, and
  RC-scale AEB thresholds (retune TTC + corridor width for ~0.2–0.5 m distances).

No changes to `intent/`, `decision/`, or `pipeline.py` logic — only new sensor/actuator
edges and a config profile.

## 4. Compute plan (Jetson Nano 4GB)

Real-time pose+tracking+intent on a Nano is the top risk. Staged:
1. **Offboard first:** run `adas_node` on your laptop on the same ROS network; the bot
   publishes camera and receives car_cmd over WiFi. Fastest to validate behavior.
   (Note: WiFi latency will inflate the measured reaction time — that's expected.)
2. **Onboard next:** move the node onto the Nano. To hit frame rate:
   - use `use_pose:false` (already recommended) → far cheaper features + smaller model;
   - YOLO **nano** detector, lower input resolution, and TensorRT FP16 export
     (`runtime/export_onnx.py` is the starting point);
   - drop detector frame rate (e.g. detect at 5–10 Hz, track/coast in between);
   - benchmark end-to-end latency early and report FPS in telemetry.

## 5. Ego speed + distance from real sensors

- **Ego speed:** `speed = (2π·wheel_radius·Δticks) / (ticks_per_rev·Δt)`, averaged over
  the two wheels. Put wheel_radius + ticks_per_rev in config. This finally makes
  `ego_speed_mps` a measured value → TTC and reaction time are real.
- **Distance:** the front ToF gives metric range to the nearest object ahead. Use it as
  the primary `distance_m` for the in-path pedestrian; fall back to monocular when the
  target is outside the ToF's narrow cone. Feed ToF distance into the per-track Kalman
  the same way fusion range was.

## 6. Brake mapping (staged rollout)

- **Phase A — warn only (default first):** on `FCW`/braking, flash the LEDs red and mark
  BRAKE on the dashboard, but keep the normal drive command. Validates detection +
  decision with zero risk of the bot lurching.
- **Phase B — motor stop:** map `PARTIAL_AEB`→reduce speed, `FULL_AEB`→zero/negative
  car_cmd (stop). Keep an imminent-collision safety net. Only enable after Phase A looks
  right on the telemetry log.

## 7. Bring-up checklist (order matters)

1. Decide the pedestrian target (§1) and retrain the intent model with `use_pose:false`
   (`data/prepare_*.py` + `intent/train.py` + `calibrate.py` + `evaluate.py`); confirm
   test-split metrics are acceptable BEFORE touching hardware.
2. `dts` environment working; `rostopic list` to capture the real topic names.
3. Build `DuckiebotSource` + a no-op actuator; run offboard in **warn-only**; watch the
   dashboard/telemetry while walking the figure across the bot's path.
4. Calibrate: wheel_radius/ticks_per_rev (drive a known distance, check speed), ToF
   sanity (place object at known range), retune AEB TTC + corridor for RC scale.
5. Switch to the Duckiebot actuator, still **warn-only**, confirm LEDs fire correctly.
6. Enable **motor-stop** (Phase B) at low speed in a controlled lane.
7. Port node onboard the Nano; benchmark FPS/latency; apply the compute mitigations.

## 8. What to confirm with me / decide
- Pedestrian target (§1) — figure + `use_pose:false` is my recommendation.
- Offboard-first vs straight-to-Nano.
- Whether to expose the PyQt dashboard remotely (offboard) or just log telemetry + use
  LEDs on-bot.

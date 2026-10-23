# HARDWARE_DUCKIEBOT.md — Live pedestrian-AEB on the Duckiebot

This is the hardware bring-up for the camera path: the Duckiebot's **webcam (mounted
~5 ft, dashcam height)** streams over WiFi to your **laptop**, the laptop runs the
point-to-point pedestrian perception, and it sends **cruise / brake / accelerate**
commands back to the Duckiebot over the **Duckietown ROS stack**.

> **Safety first.** It starts in **dry-run** (no motion) — it only prints and overlays
> decisions. Verify the brake/accelerate logic looks right, then enable motion. Keep the
> Duckiebot's shutdown button within reach whenever motion is on.

## Architecture

```
Duckiebot webcam (5 ft)  +  front ToF range sensor
   │  (WiFi: ROS camera topic / stream  +  ROS Range topic)
   ▼
LAPTOP — two braking layers:
   LAYER 1 (smart):  DuckiebotCameraSource → YOLO-pose detect+track (point-to-point)
                     → hazard gate (person in path & close) → cruise/brake/resume policy
   LAYER 2 (safety): ToF range → force STOP under a hard distance, independent of LAYER 1
   │  (WiFi: ROS Twist2DStamped on the car-command topic)
   ▼
Duckiebot motors  (cruise forward / brake-stop / resume)
```

### Two-layer braking (camera + ToF fusion)
- **Layer 1 — camera (smart):** detects a *pedestrian in the bot's path* and brakes. This
  is the pedestrian-aware decision, but it can miss / be late / mis-classify.
- **Layer 2 — ToF range (safety net):** the front time-of-flight sensor gives a direct
  distance to the nearest object ahead. If **anything** comes within
  `duckiebot.tof.safety_stop_distance_m` (default 30 cm), the bot is **forced to a stop**
  regardless of what the camera decided. It can only slow/stop, never speed up, so it's
  strictly a safety addition. This is what catches the "camera failed to stop in time"
  case — and also non-pedestrian obstacles.
- The ToF sits low on the bot and measures the nearest thing straight ahead at its own
  height, not specifically the tracked pedestrian — which is exactly why it's a fallback,
  not the primary trigger.

New files (all behind the existing abstractions — nothing in `intent/`, `decision/`,
`pipeline.py` changed):
- `sensors/duckiebot_source.py` — WiFi camera feed → `SensorFrame` (ROS or stream URL).
- `sensors/tof_range.py` — front ToF Range reader over ROS (validated distance in m).
- `hardware/actuator.py` — `ActuatorInterface`; `NoOpActuator` (dry-run) + `DuckiebotROSActuator`.
- `control/cruise_policy.py` — cruise/brake/resume state machine (unit-tested).
- `control/hazard.py` — detection-first person-in-path hazard gate (intent hook for later).
- `control/safety_stop.py` — ToF safety-net override (unit-tested).
- `run_car.py` — the live loop entry point (fuses both layers).
- `config.duckiebot.*` — all device-specific settings.

## 1. Network: put the laptop on the Duckiebot's ROS

The laptop must reach the Duckiebot's ROS master. Typical Duckietown setup:
```bash
# on the laptop, same WiFi as the bot
export ROS_MASTER_URI=http://<duckiebot-hostname>.local:11311
export ROS_IP=<your-laptop-ip>
# confirm you can see the bot's topics:
rostopic list | grep <duckiebot-hostname>
```
Note the EXACT names of the camera and car-command topics — they're namespaced by the
robot hostname and can vary by Duckietown software version. Put them in `config.yaml`
(`duckiebot.camera.ros_topic`, `duckiebot.command.car_cmd_topic`); `{veh}` is replaced
with `duckiebot.vehicle_name`.

Common names (verify on your bot):
- camera:  `/<veh>/camera_node/image/compressed`
- command: `/<veh>/car_cmd_switch_node/cmd`  (message type `duckietown_msgs/Twist2DStamped`)
- ToF range: `/<veh>/front_center_tof_driver_node/range`  (message type `sensor_msgs/Range`, metres)
  Set `duckiebot.tof.unit: cm` if your driver reports centimetres instead of metres.
  Tune `duckiebot.tof.safety_stop_distance_m` (hard stop distance) and `max_valid_m`
  (ignore readings beyond this as out-of-range/spurious).

If you'd rather not join the ROS network for the camera, set
`duckiebot.camera.source: stream` and point `stream_url` at an MJPEG/RTSP URL — the
command path still uses ROS.

## 2. Laptop environment

```bash
pip install -r requirements.txt            # torch (GPU build), ultralytics, opencv
# ROS: use the Duckietown/ROS Python env that provides rospy + duckietown_msgs
# (run_car.py / the ROS modules import them lazily; the stream+dry-run path works
#  without ROS for a first camera test).
```

## 3. Calibrate (do this before trusting distances)

- **`monocular.focal_px`** — the in-path + distance gate depends on it. Calibrate for the
  webcam: place a person of known height at a known distance, or use a checkerboard. With
  a wrong focal, the bot will brake too early or too late.
- **`duckiebot.hazard.corridor_margin_m`** — how wide the "in path" lane is (meters of
  lateral offset). Widen/narrow to match where you want it to react.
- **`duckiebot.hazard.trigger_distance_m`** and **`min_bbox_height_frac`** — how close a
  person must be before it counts. Tune on the dry-run overlay.
- **`duckiebot.control.cruise_speed_mps`** — keep low for first runs.

## 4. Run — staged rollout

```bash
# A) camera only, dry-run: confirm detection + decisions on the overlay (NO motion)
python run_car.py --veh <duckiebot-hostname> --show            # dry_run defaults true
#   walk in/out of the bot's path; watch it switch CRUISE <-> BRAKE in the banner/log.

# B) tune thresholds in config.yaml until braking triggers where you want.

# C) enable motion (LOW speed, clear area, hand on shutdown):
python run_car.py --veh <duckiebot-hostname> --drive --show

# stream-mode camera instead of ROS camera:
python run_car.py --veh <host> --source stream --url http://<bot-ip>:8080/stream --drive
```

What you'll see each frame: `CRUISE` while clear, `BRAKE` (v=0) when a pedestrian is in
path and close, `CRUISE`/accelerate again once clear — with the measured hazard→brake
**reaction time** in ms.

## 5. Adding the trained intent model (after the detection loop works)

Detection-first is the default (`hazard.use_intent_model: false`) — brake on any person
in path. To gate braking on the trained **crossing** prediction instead, set
`duckiebot.hazard.use_intent_model: true` (needs `weights/intent_model.pth`). The loop
then only brakes when the model also predicts *crossing* (prob ≥ `intent_prob_gate`).
Caveat: the model was trained on JAAD (street dashcam); expect to re-check/retune it for
your indoor Duckietown scene before relying on it.

## Safety notes
- Always start dry-run; enable `--drive` only after the overlay/logs look right.
- `cruise_speed_mps` low; test in a clear, bounded area.
- The loop commands a **stop** on exit, on Ctrl-C, and should on a stale feed
  (watchdog) — keep the physical shutdown button reachable regardless.
- This is an experimental testbed, not a certified safety system.

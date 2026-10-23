# Build Prompt — Traffic-Light Perception + Signal-Aware Braking + Interactive Dashboard

> Paste this into Claude Code opened on the `PIDS-Capstone` project. It extends the
> existing ADAS pedestrian pipeline; it does NOT start a new project.

---

## CONTEXT (read first)

You are working inside an existing, working ADAS pedestrian-safety project. Before
writing code, read these files to load full context and follow their conventions:
`PROJECT_OVERVIEW.md`, `CLAUDE.md`, `config/config.yaml`, `pipeline.py`,
`decision/aeb_policy.py`, `decision/ttc.py`, `dashboard/app.py`,
`dashboard/overlays.py`, `dashboard/telemetry.py`, `sensors/base.py`, `perception/`.

Current state: camera path works end-to-end (YOLO-pose → BoT-SORT → monocular 3D +
Kalman → temporal intent model → TTC → staged AEB → PyQt dashboard + CSV log).
Pure-python core is unit-tested; splits are clip-disjoint; model outputs raw logits.
**Scope is still SIMULATION ONLY** (datasets + demo video; ego motion and braking
simulated). Keep everything hardware-portable behind the existing `SensorFrame` /
`SensorSource` abstraction. Act like an ADAS engineer: propose the approach, flag
risks, and stop at each phase gate for my confirmation before large code changes.

## GOAL

Add three capabilities, without regressing the pedestrian pipeline:

1. **Traffic-light perception** — detect traffic lights and classify state
   (RED / YELLOW / GREEN / UNKNOWN), with distance, from a reasonable range.
2. **Signal-aware decision fusion** — combine the light state with pedestrian AEB
   using the exact policy below.
3. **Interactive dashboard** — user controls + a Slow/Stop/Go HUD element.

## DESIRED BEHAVIOR — DECISION POLICY (implement exactly, then let me tune thresholds)

Separate two distinct concerns and fuse them; do NOT collapse them into one rule:

**A. Signal compliance (normal driving intent), from light color state:**
- RED    → driving intent = STOP   (HUD: "STOP")
- YELLOW → driving intent = SLOW   (HUD: "SLOW")
- GREEN  → driving intent = GO     (HUD: "GO")
- UNKNOWN/none detected → proceed with caution (treat as GO but conservative)

**B. Pedestrian AEB (safety override), gated by driving intent:**
- The system must NOT raise an emergency brake event *merely because the light is red* —
  on red the vehicle is already stopping for the signal, and pedestrians crossing is
  expected.
- On **GREEN** (vehicle intends to proceed): if any pedestrian is in the ego path AND
  is moving / has high crossing probability, ENGAGE braking (this overrides GO).
- Keep pedestrian AEB as an ultimate safety net in all states (if a collision is truly
  imminent, brake regardless), but the *primary* pedestrian-brake trigger is the
  green-light-with-moving-pedestrian case above. Make this precedence explicit and
  configurable.

Represent the final output as a combined command with a clear reason string, e.g.
`{driving_intent: GO|SLOW|STOP, brake: bool, source: "signal"|"pedestrian_aeb"|"both",
reason: "..."}`. The HUD Slow/Stop/Go reflects driving intent; the existing
BRAKE/WARNING banner reflects pedestrian AEB. Both must be visible simultaneously.

## TRAFFIC-LIGHT PERCEPTION — APPROACH (CAMERA-ONLY; validate with me first)

**This module is CAMERA-ONLY. Do NOT use radar or LIDAR here.** Traffic-light state
(color) and the countdown timer (digits) are photometric/semantic information that
radar and LIDAR physically cannot sense — they return range/reflectivity, not color or
text. All traffic-light and timer perception runs on the RGB camera frame in
`SensorFrame.image`; `SensorFrame.radar` / `SensorFrame.lidar_points` are ignored by
this module even when present (nuScenes/fusion path).

- Two-stage, lightweight (Jetson-friendly): (1) detect traffic-light boxes with a
  YOLO detector; (2) classify the illuminated state on the crop. Start with an
  HSV/brightness color classifier as a baseline, then optionally a small CNN head.
- Add `perception/traffic_light.py` returning a `TrafficLightState` dataclass:
  `{state: enum, bbox_xyxy, confidence, distance_m (monocular, reuse the height/geometry
  approach), seconds_remaining: Optional[float]}`.
- **Countdown timer (numeric):** implement as a third stage — inside the detected light
  box, find the numeric timer region, then a small CNN digit classifier (per digit;
  11 classes = 0-9 plus a "null/blank" class for a missing tens digit). This is the
  standard approach in the literature (YOLO region → CNN digit heads). It is
  **best-effort and range-limited**: only attempt it when the timer crop is large enough
  (roughly ≥40×40 px per the public datasets); otherwise return `seconds_remaining=None`
  and fall back to state-only. Do NOT let timer recognition block or slow the main loop.
  Slow/Stop/Go MUST work from color state alone.
- Temporal smoothing: debounce state across frames (a few-frame majority vote) so a
  single misclassified frame can't flip STOP↔GO. For the timer, sanity-check that the
  value is monotonically counting down and reject frames that violate it.

## DATASETS & MODELS (CAMERA-ONLY; list exact steps in SETUP.md; pause on any EULA)

Traffic-light detection + state (pick one to start; justify the choice):
- **LISA Traffic Light**, **Bosch Small Traffic Lights (BSTLD)**, **DriveU (DTLD)**,
  or **S2TLD** — standard camera traffic-light datasets with color-state labels.
- Quick baseline: the COCO `traffic light` class in the existing YOLO detector (note its
  accuracy limits), then fine-tune on one of the above for reliable state at range.

Countdown-timer digit recognition (camera-only; datasets DO exist, they're just smaller):
- **Open-source traffic-light + countdown-display detection dataset** (Science China
  Information Sciences, 2025) — ~20k-image traffic-light subset with countdown displays.
- **TSCTNet** (Traffic Signal + Countdown Timer Detection Network) — annotated countdown
  digits; good reference for the detect-region → classify-digits pipeline.
- **VT-HMM countdown-timer set** — small sequence set (double-digit displays) split into
  easy (≥40×40 px, clear) and hard (small) parts; useful for validating the range limit.
- Train the digit classifier on these; keep timer recognition optional with graceful
  fallback (see approach above). Report timer read-rate vs distance so the range limit is
  quantified, not assumed.

## INTERACTIVE DASHBOARD (extend `dashboard/app.py`, keep PyQt)

Add, without breaking the current layout or the CSV telemetry contract:
- Transport controls: Play / Pause, step one frame, a seek slider, and a playback-speed
  control.
- Overlay toggles: skeleton, trails, boxes, traffic-light box, ego corridor.
- Click a pedestrian (or select by ID) to open a side/detail panel: crossing-probability
  history (small plot), distance, TTC, in-path flag.
- Live, config-backed sliders for AEB TTC thresholds and crossing-prob gate, so I can
  tune behavior at runtime; persist edits back to `config.yaml` on request.
- New HUD element: large Slow / Stop / Go indicator driven by driving intent, shown
  alongside the existing BRAKE/WARNING banner (both visible at once). Traffic-light
  state + distance (+ timer if available) shown as text.
- Keep a headless mode equivalent (mirror `run_demo.py --headless`) that renders the
  same overlays and logs the new fields.

## INTEGRATION POINTS (respect existing architecture)

- Extend `SensorFrame` only if needed; camera image is already present.
- Run traffic-light perception inside `pipeline.py` each frame, timed with the existing
  `LatencyBudget` (add a `traffic_light` stage).
- Extend `PipelineResult` with the traffic-light state and the combined command; extend
  `dashboard/telemetry.py` FIELDS (append columns, don't reorder existing ones).
- Put all new thresholds/paths in `config/config.yaml` (no hardcoding).
- Add a new `decision/signal_policy.py` (pure functions, unit-testable) for the state
  machine above; add `tests/test_signal_policy.py` covering: red-no-brake, green+moving-
  pedestrian-brake, yellow-slow, unknown-caution, and the imminent-collision safety net.

## METRICS / ACCEPTANCE

- Traffic-light: state-classification accuracy + detection range (at what distance state
  becomes reliable); timer read-rate vs distance (expect it to drop off with range).
- Decision fusion: verify with unit tests that red does NOT trigger pedestrian AEB, green
  + in-path moving pedestrian DOES, and the safety net still fires on imminent collision.
- No regression in the existing pedestrian metrics or dashboard/CSV behavior.

## PHASE GATES (stop for my confirmation at each)

1. Proposal: traffic-light dataset/model choice, `TrafficLightState` schema,
   `signal_policy` state machine, and the dashboard interaction list. No large code yet.
2. `perception/traffic_light.py` + `decision/signal_policy.py` + unit tests.
3. Pipeline + telemetry integration (headless verified on the demo video).
4. Interactive dashboard + Slow/Stop/Go HUD.
5. Optional best-effort countdown-timer OCR with fallback.

## FIRST TASK

Restate the decision policy in your own words to confirm you understand the
red-vs-green pedestrian-braking distinction, then present the Phase-1 proposal (dataset
choice with trade-offs, the two new schemas, and the dashboard interaction list) and
wait for my go-ahead. Flag the top risks — especially real-time cost of adding a second
detector, and the timer-OCR range limitation.

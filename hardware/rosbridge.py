"""
rosbridge link — run the whole pipeline from the laptop's (torch) venv without a
native ROS install, by talking to the Duckiebot over a rosbridge websocket.

Why this exists: on a Duckietown VM, ROS 1 Noetic lives inside the `dts` Docker
containers, not in the host Python where torch/ultralytics are installed. `roslibpy`
is a pure-Python ROS client (pip-installable) that connects to `rosbridge_websocket`
on the robot, so the same venv that runs YOLO can also read the camera + ToF and
publish car commands.

One `RosbridgeLink` owns the websocket; thin adapters expose it as the project's
`SensorSource` (camera + ToF) and `ActuatorInterface` (car_cmd), so run_car.py's loop
is unchanged. roslibpy is imported lazily.

Enable with `--bridge` (or config.duckiebot.bridge.enabled: true). Requires
rosbridge_websocket running on the bot (Duckietown dashboards use it; see
HARDWARE_DUCKIEBOT.md for how to confirm/launch it).
"""
from __future__ import annotations

import base64
import threading
import time
from typing import Iterator, Optional

import numpy as np

from sensors.base import SensorFrame, SensorSource
from hardware.actuator import ActuatorInterface
from control.safety_stop import SafetyConfig, normalize_range, is_valid


def _fill(tmpl: str, veh: str) -> str:
    return tmpl.replace("{veh}", veh)


class RosbridgeLink:
    """Owns the roslibpy websocket + camera/ToF subscriptions + car_cmd publisher."""

    def __init__(self, cfg: dict):
        import roslibpy                                   # lazy
        self._roslibpy = roslibpy
        d = cfg["duckiebot"]
        b = d.get("bridge", {})
        veh = d["vehicle_name"]
        host = b.get("host") or f"{veh}.local"
        port = int(b.get("port", 9090))

        self._ros = roslibpy.Ros(host=host, port=port)
        self._ros.run()                                  # connects in a background thread
        t0 = time.time()
        while not self._ros.is_connected and time.time() - t0 < 10:
            time.sleep(0.1)
        if not self._ros.is_connected:
            raise ConnectionError(
                f"Could not connect to rosbridge at ws://{host}:{port}. "
                f"Is rosbridge_websocket running on the bot? (see HARDWARE_DUCKIEBOT.md)")

        self._lock = threading.Lock()
        self._img: Optional[np.ndarray] = None
        self._tof_m: Optional[float] = None
        self._safety = SafetyConfig.from_config(cfg)
        self._fps = float(d["camera"].get("fps", 30))

        # camera
        cam_topic = _fill(d["camera"]["ros_topic"], veh)
        self._cam = roslibpy.Topic(self._ros, cam_topic, "sensor_msgs/CompressedImage",
                                   queue_length=1, throttle_rate=int(1000 / self._fps))
        self._cam.subscribe(self._on_image)

        # ToF (optional)
        self._tof_topic = None
        if d.get("tof", {}).get("enabled", False):
            tof_topic = _fill(d["tof"]["ros_topic"], veh)
            self._tof_topic = roslibpy.Topic(self._ros, tof_topic, "sensor_msgs/Range",
                                             queue_length=1)
            self._tof_topic.subscribe(self._on_tof)

        # car command publisher
        cmd_topic = _fill(d["command"]["car_cmd_topic"], veh)
        self._cmd = roslibpy.Topic(self._ros, cmd_topic, "duckietown_msgs/Twist2DStamped")
        self._cmd.advertise()

    # ── subscriptions ─────────────────────────────────────────────────────────
    def _on_image(self, msg):
        import cv2
        try:
            raw = base64.b64decode(msg["data"])          # rosbridge sends bytes as base64
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is not None:
            with self._lock:
                self._img = img

    def _on_tof(self, msg):
        m = normalize_range(msg.get("range"), self._safety.unit)
        with self._lock:
            self._tof_m = m if is_valid(m, self._safety) else None

    # ── accessors used by the adapters ────────────────────────────────────────
    def grab_image(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._img is None else self._img.copy()

    def read_tof(self) -> Optional[float]:
        with self._lock:
            return self._tof_m

    @property
    def fps(self) -> float:
        return self._fps

    # ── actuator side ─────────────────────────────────────────────────────────
    def drive(self, v: float, omega: float) -> None:
        self._cmd.publish(self._roslibpy.Message({"v": float(v), "omega": float(omega)}))

    def close(self) -> None:
        try:
            self.drive(0.0, 0.0)                         # leave the bot stopped
            self._cmd.unadvertise()
            self._cam.unsubscribe()
            if self._tof_topic is not None:
                self._tof_topic.unsubscribe()
        finally:
            self._ros.terminate()


class BridgeCameraSource(SensorSource):
    """SensorSource backed by the rosbridge camera (+ToF attached per frame)."""

    def __init__(self, link: RosbridgeLink):
        self._link = link
        self._idx = 0
        self._ego = 0.0

    @property
    def fps(self) -> float:
        return self._link.fps

    def set_ego_speed(self, v: float) -> None:
        self._ego = v

    def __iter__(self) -> Iterator[SensorFrame]:
        period = 1.0 / max(self._link.fps, 1e-6)
        while True:
            img = self._link.grab_image()
            if img is not None:
                yield SensorFrame(frame_idx=self._idx, timestamp_s=time.time(),
                                  image=img, ego_speed_mps=self._ego,
                                  tof_distance_m=self._link.read_tof())
                self._idx += 1
            time.sleep(period)

    def close(self) -> None:
        pass          # the link is closed by run_car via the actuator


class BridgeToF:
    """ToF reader adapter (matches sensors.tof_range.ToFRangeReader.read())."""
    def __init__(self, link: RosbridgeLink):
        self._link = link

    def read(self) -> Optional[float]:
        return self._link.read_tof()


class BridgeActuator(ActuatorInterface):
    def __init__(self, link: RosbridgeLink):
        self._link = link

    def drive(self, v: float, omega: float) -> None:
        self._link.drive(v, omega)

    def close(self) -> None:
        self.stop()
        self._link.close()


def build_bridge(cfg: dict):
    """Create the link + the three adapters run_car.py needs."""
    link = RosbridgeLink(cfg)
    return BridgeCameraSource(link), BridgeToF(link), BridgeActuator(link)

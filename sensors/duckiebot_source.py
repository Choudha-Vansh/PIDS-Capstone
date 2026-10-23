"""
Live camera source for the Duckiebot, streamed over WiFi to the laptop.

Two ingest modes (set in config.duckiebot.camera.source):
  - "ros"    : subscribe to the Duckiebot's compressed-image topic via rospy and
               decode with OpenCV. Needs the laptop on the Duckietown ROS network.
  - "stream" : read an MJPEG/RTSP URL with cv2.VideoCapture (simplest; works if the
               bot exposes an HTTP stream).

Both yield the SAME SensorFrame the rest of the pipeline already consumes, so nothing
downstream changes. ROS imports are lazy so the module (and tests) load without ROS.
"""
from __future__ import annotations

import threading
import time
from typing import Iterator, Optional

import numpy as np

from sensors.base import SensorFrame, SensorSource


def _fill_topic(tmpl: str, veh: str) -> str:
    return tmpl.replace("{veh}", veh)


class DuckiebotCameraSource(SensorSource):
    def __init__(self, cfg: dict):
        d = cfg["duckiebot"]
        self.veh = d["vehicle_name"]
        cam = d["camera"]
        self.mode = cam.get("source", "ros")
        self._fps = float(cam.get("fps", 30))
        self._ego_speed = float(d["control"].get("cruise_speed_mps", 0.0))
        self._idx = 0

        if self.mode == "stream":
            import cv2
            url = cam.get("stream_url", "")
            if not url:
                raise ValueError("duckiebot.camera.stream_url is empty but source=stream")
            self._cap = cv2.VideoCapture(url)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open Duckiebot stream: {url}")
            r = self._cap.get(cv2.CAP_PROP_FPS)
            if r and r > 1:
                self._fps = r
        elif self.mode == "ros":
            self._setup_ros(_fill_topic(cam["ros_topic"], self.veh))
        else:
            raise ValueError(f"unknown camera source: {self.mode}")

    # ── ROS mode ──────────────────────────────────────────────────────────────
    def _setup_ros(self, topic: str):
        import rospy                                   # lazy
        from sensor_msgs.msg import CompressedImage
        self._rospy = rospy
        self._latest = None
        self._lock = threading.Lock()
        if not rospy.core.is_initialized():
            rospy.init_node("adas_laptop_camera", anonymous=True, disable_signals=True)
        self._sub = rospy.Subscriber(topic, CompressedImage, self._on_image,
                                     queue_size=1, buff_size=2 ** 24)

    def _on_image(self, msg):
        import cv2
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            with self._lock:
                self._latest = img

    # ── common ────────────────────────────────────────────────────────────────
    @property
    def fps(self) -> float:
        return self._fps

    def set_ego_speed(self, v: float) -> None:
        """Let the control loop report the current commanded speed as ego speed."""
        self._ego_speed = v

    def __iter__(self) -> Iterator[SensorFrame]:
        if self.mode == "stream":
            yield from self._iter_stream()
        else:
            yield from self._iter_ros()

    def _iter_stream(self):
        period = 1.0 / max(self._fps, 1e-6)
        while True:
            ok, img = self._cap.read()
            if not ok:
                break
            yield self._make_frame(img)
            time.sleep(0)  # yield; VideoCapture paces itself

    def _iter_ros(self):
        rate = self._rospy.Rate(self._fps)
        while not self._rospy.is_shutdown():
            with self._lock:
                img = None if self._latest is None else self._latest.copy()
            if img is not None:
                yield self._make_frame(img)
            rate.sleep()

    def _make_frame(self, img) -> SensorFrame:
        f = SensorFrame(frame_idx=self._idx, timestamp_s=time.time(),
                        image=img, ego_speed_mps=self._ego_speed)
        self._idx += 1
        return f

    def close(self) -> None:
        if self.mode == "stream" and getattr(self, "_cap", None) is not None:
            self._cap.release()

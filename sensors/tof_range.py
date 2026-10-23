"""
Front time-of-flight range reader for the Duckiebot, over the Duckietown ROS stack.

Subscribes to the ToF Range topic and exposes the latest VALIDATED distance in metres
(or None when there's no trustworthy reading). The validation / stop logic lives in
control/safety_stop.py (pure + tested); this file is just the ROS plumbing, so rospy is
imported lazily and the rest of the project loads without ROS.

Works regardless of the camera ingest mode (the command path is ROS anyway).
"""
from __future__ import annotations

import threading
from typing import Optional

from control.safety_stop import SafetyConfig, normalize_range, is_valid


class ToFRangeReader:
    def __init__(self, cfg: dict):
        d = cfg["duckiebot"]
        t = d.get("tof", {})
        self.enabled = t.get("enabled", False)
        self.safety = SafetyConfig.from_config(cfg)
        self._latest_m: Optional[float] = None
        self._lock = threading.Lock()
        self._sub = None

        if not self.enabled:
            return

        import rospy                                   # lazy
        from sensor_msgs.msg import Range
        topic = t["ros_topic"].replace("{veh}", d["vehicle_name"])
        if not rospy.core.is_initialized():
            rospy.init_node("adas_laptop_tof", anonymous=True, disable_signals=True)
        self._sub = rospy.Subscriber(topic, Range, self._on_range, queue_size=1)
        self._topic = topic

    def _on_range(self, msg):
        # sensor_msgs/Range.range is metres by spec; `unit` config converts if the
        # driver reports cm instead.
        m = normalize_range(msg.range, self.safety.unit)
        with self._lock:
            self._latest_m = m if is_valid(m, self.safety) else None

    def read(self) -> Optional[float]:
        """Latest validated range in metres, or None if unavailable/out-of-range."""
        if not self.enabled:
            return None
        with self._lock:
            return self._latest_m

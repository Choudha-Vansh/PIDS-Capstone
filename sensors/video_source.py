"""
Camera-only sensor source backed by a video file (your demo clip / dashcam).
No radar or LIDAR -> downstream 3D localization falls back to monocular distance.
"""
from __future__ import annotations

from typing import Iterator, Optional

import cv2

from sensors.base import SensorFrame, SensorSource


class VideoSource(SensorSource):
    def __init__(self, path: str, ego_speed_mps: float = 0.0):
        self.path = path
        self._ego_speed = ego_speed_mps
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> Optional[int]:
        return self._total

    def __iter__(self) -> Iterator[SensorFrame]:
        idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield SensorFrame(
                frame_idx=idx,
                timestamp_s=idx / self._fps,
                image=frame,
                ego_speed_mps=self._ego_speed,
                radar=None,
                lidar_points=None,
            )
            idx += 1

    def close(self) -> None:
        self._cap.release()

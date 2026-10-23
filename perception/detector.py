"""
Pedestrian detection + pose (17 COCO keypoints) via YOLO-pose (ultralytics).
Lazy torch/ultralytics import so pure-python modules import cleanly without them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Detection:
    bbox_xyxy: np.ndarray           # (4,) pixels
    conf: float
    keypoints: np.ndarray           # (17, 3) -> x, y, visibility (pixels)
    track_id: int = -1              # filled in by the tracker

    @property
    def bbox_height_px(self) -> float:
        return float(self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def center_px(self):
        x = (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0
        y = (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2.0
        return float(x), float(y)


class PoseDetector:
    """Thin wrapper around a YOLO-pose model, person class only."""

    PERSON_CLASS = 0

    def __init__(self, model_path: str, conf: float = 0.35, device: str = None):
        from ultralytics import YOLO  # lazy
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self, image: np.ndarray) -> List[Detection]:
        res = self.model.predict(image, classes=[self.PERSON_CLASS],
                                 conf=self.conf, verbose=False,
                                 device=self.device)[0]
        dets: List[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return dets
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kpts = (res.keypoints.data.cpu().numpy()
                if res.keypoints is not None else None)
        for i in range(len(boxes)):
            kp = kpts[i] if kpts is not None else np.zeros((17, 3), np.float32)
            dets.append(Detection(bbox_xyxy=boxes[i], conf=float(confs[i]),
                                  keypoints=kp))
        return dets

"""
Multi-object tracking with stable IDs using ultralytics' built-in BoT-SORT.
BoT-SORT keeps IDs across brief occlusion far better than naive IoU tracking,
which is what caused the ID-switch / label-mismatch problems in the earlier
bounding-box-only prototype.

We run detection+tracking together via model.track() (ultralytics manages the
tracker state), then attach keypoints back onto the tracked boxes.
"""
from __future__ import annotations

from typing import List

import numpy as np

from perception.detector import Detection


class PoseTracker:
    PERSON_CLASS = 0

    def __init__(self, model_path: str, conf: float = 0.35,
                 tracker_cfg: str = "botsort.yaml", device: str = None):
        from ultralytics import YOLO  # lazy
        self.model = YOLO(model_path)
        self.conf = conf
        self.tracker_cfg = tracker_cfg
        self.device = device

    def track(self, image: np.ndarray) -> List[Detection]:
        res = self.model.track(image, classes=[self.PERSON_CLASS], conf=self.conf,
                               persist=True, verbose=False,
                               tracker=self.tracker_cfg, device=self.device)[0]
        dets: List[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return dets
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = (res.boxes.id.cpu().numpy().astype(int)
               if res.boxes.id is not None else np.full(len(boxes), -1))
        kpts = (res.keypoints.data.cpu().numpy()
                if res.keypoints is not None else None)
        for i in range(len(boxes)):
            kp = kpts[i] if kpts is not None else np.zeros((17, 3), np.float32)
            dets.append(Detection(bbox_xyxy=boxes[i], conf=float(confs[i]),
                                  keypoints=kp, track_id=int(ids[i])))
        return dets

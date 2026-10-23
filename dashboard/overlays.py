"""
OpenCV drawing helpers for the dashboard video pane: boxes, skeletons, trails,
distance labels, colored by AEB severity. Pure OpenCV/numpy (no Qt), so it can be
unit-visualized headless too.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict

import cv2
import numpy as np

from decision.aeb_policy import AEBLevel

# COCO-17 skeleton edges
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]

LEVEL_COLOR = {
    AEBLevel.NONE: (0, 200, 0),        # green
    AEBLevel.FCW: (0, 200, 255),       # amber
    AEBLevel.PARTIAL_AEB: (0, 100, 255),  # orange
    AEBLevel.FULL_AEB: (0, 0, 255),    # red
}


class TrailStore:
    def __init__(self, length: int = 40):
        self.length = length
        self.trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=length))

    def push(self, tid, center):
        self.trails[tid].append((int(center[0]), int(center[1])))

    def draw(self, frame, tid, color):
        pts = list(self.trails[tid])
        for i in range(1, len(pts)):
            a = i / len(pts)
            c = tuple(int(v * a) for v in color)
            cv2.line(frame, pts[i - 1], pts[i], c, 2)


def draw_skeleton(frame, kpts, color, conf_th=0.3):
    for (a, b) in SKELETON:
        if kpts[a, 2] > conf_th and kpts[b, 2] > conf_th:
            pa = (int(kpts[a, 0]), int(kpts[a, 1]))
            pb = (int(kpts[b, 0]), int(kpts[b, 1]))
            cv2.line(frame, pa, pb, color, 2)
    for i in range(kpts.shape[0]):
        if kpts[i, 2] > conf_th:
            cv2.circle(frame, (int(kpts[i, 0]), int(kpts[i, 1])), 3, color, -1)


def draw_track(frame, tv, trails: TrailStore, show_skeleton=True, show_trails=True):
    color = LEVEL_COLOR[tv.decision.level]
    x0, y0, x1, y1 = [int(v) for v in tv.bbox_xyxy]
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

    if show_trails:
        trails.push(tv.track_id, tv.center_px)
        trails.draw(frame, tv.track_id, color)
    if show_skeleton:
        draw_skeleton(frame, tv.keypoints, color)

    label = f"ID{tv.track_id}  {tv.crossing_prob:.0%}  {tv.distance_m:.1f}m"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x0, y0 - th - 6), (x0 + tw + 4, y0), color, -1)
    cv2.putText(frame, label, (x0 + 2, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    return frame

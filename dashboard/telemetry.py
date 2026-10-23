"""
CSV telemetry logging shared between the windowed dashboard and the headless
run_demo.py path, so the two can't drift into different column formats.
"""
from __future__ import annotations

import csv
import os

FIELDS = ["frame", "level", "ttc_s", "distance_m", "crossing_prob",
          "ego_speed_mps", "total_latency_ms", "reaction_ms", "stop_avoided"]


class TelemetryLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._f = open(path, "w", newline="")
        self._csv = csv.writer(self._f)
        self._csv.writerow(FIELDS)

    def log(self, r):
        w = r.worst
        self._csv.writerow([
            r.frame_idx,
            (w.decision.level.name if w else "NONE"),
            (f"{w.decision.ttc_s:.3f}" if w else ""),
            (f"{w.distance_m:.2f}" if w else ""),
            (f"{w.crossing_prob:.3f}" if w else ""),
            f"{r.ego_speed_mps:.2f}",
            f"{r.latency.total_ms:.2f}",
            ("" if r.reaction_ms is None else f"{r.reaction_ms:.1f}"),
            ("" if r.stop_avoided is None else int(r.stop_avoided)),
        ])

    def close(self):
        self._f.close()

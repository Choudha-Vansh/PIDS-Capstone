"""
PyQt5 dashboard: video pane (top) with overlays + telemetry/warning panel (bottom).

Panel shows, for the highest-severity pedestrian this frame:
  - big BRAKE / WARNING / CLEAR banner (color-coded)
  - TTC (s), distance (m), crossing probability, ego speed
  - per-stage + total pipeline latency (ms)
  - measured system reaction time (ms) and whether the simulated stop avoids collision

Also logs every frame's telemetry to a CSV for offline analysis.

Run via run_demo.py (which builds the pipeline and hands frames to this window).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QFrame)

from decision.aeb_policy import AEBLevel, LEVEL_LABEL
from dashboard.overlays import TrailStore, draw_track, LEVEL_COLOR
from dashboard.telemetry import TelemetryLogger

BANNER_QCOLOR = {
    AEBLevel.NONE: "#1a9e46",
    AEBLevel.FCW: "#e0a000",
    AEBLevel.PARTIAL_AEB: "#e06a00",
    AEBLevel.FULL_AEB: "#d0021b",
}


class Dashboard(QWidget):
    def __init__(self, source, pipeline, cfg):
        super().__init__()
        self.source = source
        self.pipeline = pipeline
        self.cfg = cfg
        dc = cfg["dashboard"]
        self.trails = TrailStore(dc.get("trail_length", 40))
        self.show_skel = dc.get("show_skeleton", True)
        self.show_trails = dc.get("show_trails", True)
        self._frame_iter = iter(source)

        self.setWindowTitle(dc.get("window_title", "ADAS Dashboard"))
        self._build_ui()

        self.log_path = dc.get("export_log", "results/session_log.csv")
        self._telemetry = TelemetryLogger(self.log_path)

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000 / max(source.fps, 1)))

    def _build_ui(self):
        root = QVBoxLayout(self)
        self.video = QLabel(); self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(960, 540)
        self.video.setStyleSheet("background:#000;")
        root.addWidget(self.video, stretch=5)

        panel = QFrame(); panel.setStyleSheet("background:#111;color:#eee;")
        pl = QHBoxLayout(panel)

        self.banner = QLabel("CLEAR"); self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setFont(QFont("Arial", 34, QFont.Bold))
        self.banner.setFixedWidth(320)
        self.banner.setStyleSheet(f"background:{BANNER_QCOLOR[AEBLevel.NONE]};"
                                  f"color:white;border-radius:8px;")
        pl.addWidget(self.banner)

        grid = QGridLayout()
        self.fields = {}
        rows = [("TTC (s)", "ttc"), ("Distance (m)", "dist"),
                ("Crossing prob", "prob"), ("Ego speed (m/s)", "speed"),
                ("Latency (ms)", "lat"), ("Reaction (ms)", "react"),
                ("Stop avoided", "stop"), ("Frame", "frame")]
        for i, (label, key) in enumerate(rows):
            cap = QLabel(label); cap.setStyleSheet("color:#888;")
            val = QLabel("-"); val.setFont(QFont("Consolas", 16, QFont.Bold))
            grid.addWidget(cap, i % 4, (i // 4) * 2)
            grid.addWidget(val, i % 4, (i // 4) * 2 + 1)
            self.fields[key] = val
        pl.addLayout(grid, stretch=1)
        root.addWidget(panel, stretch=2)

    def _tick(self):
        try:
            frame = next(self._frame_iter)
        except StopIteration:
            self.timer.stop(); self._finish(); return

        result = self.pipeline.process(frame)
        img = frame.image.copy()
        for tv in result.tracks:
            draw_track(img, tv, self.trails, self.show_skel, self.show_trails)
        cv2.putText(img, f"Frame {result.frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self._show(img)
        self._update_panel(result)
        self._log_row(result)

    def _update_panel(self, r):
        w = r.worst
        level = w.decision.level if w else AEBLevel.NONE
        self.banner.setText(LEVEL_LABEL[level])
        self.banner.setStyleSheet(f"background:{BANNER_QCOLOR[level]};"
                                  f"color:white;border-radius:8px;")
        f = self.fields
        f["frame"].setText(str(r.frame_idx))
        f["speed"].setText(f"{r.ego_speed_mps:.1f}")
        f["lat"].setText(f"{r.latency.total_ms:.1f}")
        f["react"].setText("-" if r.reaction_ms is None else f"{r.reaction_ms:.0f}")
        f["stop"].setText("-" if r.stop_avoided is None
                          else ("YES" if r.stop_avoided else "NO"))
        if w:
            ttc = w.decision.ttc_s
            f["ttc"].setText("inf" if ttc == float("inf") else f"{ttc:.2f}")
            f["dist"].setText(f"{w.distance_m:.1f}")
            f["prob"].setText(f"{w.crossing_prob:.0%}")
        else:
            for k in ("ttc", "dist", "prob"):
                f[k].setText("-")

    def _log_row(self, r):
        self._telemetry.log(r)

    def _show(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video.width(), self.video.height(), Qt.KeepAspectRatio))

    def _finish(self):
        self._telemetry.close()
        summary = self.pipeline.reaction.summary()
        print(f"\nSession done. Reaction-time summary: {summary}")
        print(f"Telemetry log -> {self.log_path}")

    def closeEvent(self, e):
        try:
            self._telemetry.close()
        except Exception:
            pass
        self.source.close()
        e.accept()


def run(source, pipeline, cfg):
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    win = Dashboard(source, pipeline, cfg)
    win.resize(1000, 800)
    win.show()
    app.exec_()

"""
Latency instrumentation. This is how 'system reaction time' becomes a *measured*
number instead of a guess: every stage stamps how long it took, and the decision
layer records the moment a pedestrian first becomes a hazard vs the moment a brake
command is issued.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class Stopwatch:
    """Context manager that records elapsed ms into a LatencyBudget stage."""

    def __init__(self, budget: "LatencyBudget", stage: str):
        self.budget = budget
        self.stage = stage
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self.budget.record(self.stage, dt_ms)
        return False


@dataclass
class LatencyBudget:
    """
    Per-frame latency across pipeline stages, in milliseconds.
    Typical stages: 'sensor', 'detect_pose', 'track', 'fusion', 'intent',
    'decision', 'render'.
    """
    stages: Dict[str, float] = field(default_factory=dict)

    def stopwatch(self, stage: str) -> Stopwatch:
        return Stopwatch(self, stage)

    def record(self, stage: str, ms: float) -> None:
        self.stages[stage] = ms

    @property
    def total_ms(self) -> float:
        return float(sum(self.stages.values()))

    def as_row(self) -> Dict[str, float]:
        row = dict(self.stages)
        row["total_ms"] = self.total_ms
        return row


@dataclass
class ReactionTimer:
    """
    Measures end-to-end reaction time: from the frame a pedestrian first crosses
    the hazard threshold (crossing prob high AND TTC below FCW) to the frame a
    brake command (partial/full AEB) is actually issued.
    """
    fps: float = 30.0
    _hazard_started_frame: Optional[int] = None
    reactions_ms: List[float] = field(default_factory=list)

    def note_hazard(self, frame_idx: int) -> None:
        """Call when a pedestrian first becomes a potential hazard."""
        if self._hazard_started_frame is None:
            self._hazard_started_frame = frame_idx

    def note_brake(self, frame_idx: int, pipeline_latency_ms: float = 0.0) -> Optional[float]:
        """
        Call when a brake command is issued. Returns the reaction time in ms
        (perception+decision frames elapsed since hazard onset, plus this frame's
        pipeline latency), or None if there was no pending hazard.
        """
        if self._hazard_started_frame is None:
            return None
        frames_elapsed = frame_idx - self._hazard_started_frame
        rt_ms = frames_elapsed * (1000.0 / self.fps) + pipeline_latency_ms
        self.reactions_ms.append(rt_ms)
        self._hazard_started_frame = None
        return rt_ms

    def clear(self) -> None:
        """Hazard resolved without braking (e.g. pedestrian stopped)."""
        self._hazard_started_frame = None

    def summary(self) -> Dict[str, float]:
        if not self.reactions_ms:
            return {"count": 0, "mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        s = sorted(self.reactions_ms)
        p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
        return {
            "count": len(s),
            "mean_ms": sum(s) / len(s),
            "p95_ms": p95,
            "max_ms": s[-1],
        }

"""Unit tests for the decision + dynamics math (no torch/GPU needed)."""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from decision.ttc import time_to_collision, is_in_conflict
from decision.aeb_policy import AEBLevel, AEBThresholds, decide
from decision.vehicle_dynamics import VehicleParams, simulate_stop
from common.geometry import (
    monocular_distance_from_height, closing_speed_mps, lateral_offset_to_path
)
from common.latency import LatencyBudget, ReactionTimer


TH = AEBThresholds()


def test_ttc_basic():
    assert time_to_collision(20.0, 10.0) == 2.0
    assert math.isinf(time_to_collision(20.0, 0.0))       # not closing
    assert time_to_collision(0.0, 10.0) == 0.0            # already there


def test_conflict():
    assert is_in_conflict(0.0) is True
    assert is_in_conflict(0.2) is True
    assert is_in_conflict(1.5) is False


def test_decide_escalation():
    # critical TTC, high crossing prob, in corridor -> full brake
    d = decide(0.5, 0.9, in_corridor=True, thresholds=TH)
    assert d.level == AEBLevel.FULL_AEB and d.is_braking

    # medium TTC -> partial
    d = decide(1.2, 0.8, in_corridor=True, thresholds=TH)
    assert d.level == AEBLevel.PARTIAL_AEB

    # far TTC -> warning only
    d = decide(2.3, 0.6, in_corridor=True, thresholds=TH)
    assert d.level == AEBLevel.FCW and not d.is_braking

    # low crossing prob -> no braking even if TTC is low
    d = decide(0.5, 0.1, in_corridor=True, thresholds=TH)
    assert not d.is_braking

    # off corridor -> no braking (warning at most)
    d = decide(0.5, 0.9, in_corridor=False, thresholds=TH)
    assert not d.is_braking


def test_stop_avoided():
    p = VehicleParams()
    # 10 m/s (36 km/h), 30 m gap, full brake -> should stop in time
    r = simulate_stop(10.0, 30.0, AEBLevel.FULL_AEB, p)
    assert r.collision_avoided and r.impact_speed_mps == 0.0


def test_stop_not_avoided():
    p = VehicleParams()
    # 20 m/s (72 km/h), 5 m gap -> cannot stop, impact speed > 0
    r = simulate_stop(20.0, 5.0, AEBLevel.FULL_AEB, p)
    assert not r.collision_avoided and r.impact_speed_mps > 0.0


def test_no_brake_means_collision():
    p = VehicleParams()
    r = simulate_stop(15.0, 40.0, AEBLevel.FCW, p)   # warning only, no decel
    assert not r.collision_avoided and math.isinf(r.stopping_distance_m)


def test_monocular_distance():
    # a 1.7 m pedestrian at focal 1000 with 170 px box -> 10 m
    d = monocular_distance_from_height(170.0, 1000.0, 1.7)
    assert abs(d - 10.0) < 1e-6


def test_closing_speed():
    assert closing_speed_mps(20.0, 18.0, 0.1) == 20.0    # closing 2m in 0.1s
    assert closing_speed_mps(18.0, 20.0, 0.1) == 0.0     # opening -> clamped 0


def test_lateral_offset():
    assert lateral_offset_to_path(0.5, 1.2) == 0.0       # inside corridor
    assert abs(lateral_offset_to_path(3.0, 1.2) - 1.8) < 1e-9


def test_reaction_timer():
    rt = ReactionTimer(fps=10.0)
    rt.note_hazard(frame_idx=100)
    val = rt.note_brake(frame_idx=103, pipeline_latency_ms=5.0)
    # 3 frames at 10 fps = 300 ms + 5 ms pipeline
    assert abs(val - 305.0) < 1e-6
    assert rt.summary()["count"] == 1


def test_latency_budget():
    b = LatencyBudget()
    b.record("detect_pose", 12.0)
    b.record("intent", 3.0)
    assert b.total_ms == 15.0
    assert b.as_row()["total_ms"] == 15.0


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")

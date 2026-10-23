"""Tests for the ToF range safety net (pure python)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from control.cruise_policy import DriveCommand, DriveState
from control.safety_stop import (
    SafetyConfig, normalize_range, is_valid, tof_triggered, apply_safety_stop,
)

CRUISE = DriveCommand(0.22, 0.0, DriveState.CRUISE, braking=False, reason="cruise")
CFG = SafetyConfig(enabled=True, safety_stop_distance_m=0.30, max_valid_m=1.20, unit="m")


def test_normalize_units():
    assert normalize_range(0.25, "m") == 0.25
    assert abs(normalize_range(25, "cm") - 0.25) < 1e-9
    assert normalize_range(None, "cm") is None


def test_validity_gate():
    assert is_valid(0.5, CFG)
    assert not is_valid(None, CFG)
    assert not is_valid(0.0, CFG)          # zero / no return
    assert not is_valid(-1.0, CFG)         # negative
    assert not is_valid(2.0, CFG)          # beyond max_valid_m (spurious)


def test_trigger_threshold():
    assert tof_triggered(0.20, CFG)        # inside 30 cm -> stop
    assert tof_triggered(0.30, CFG)        # exactly at threshold -> stop
    assert not tof_triggered(0.35, CFG)    # just outside -> no
    assert not tof_triggered(None, CFG)
    assert not tof_triggered(0.10, SafetyConfig(enabled=False))  # disabled


def test_override_forces_stop():
    out = apply_safety_stop(CRUISE, 0.18, CFG)   # object at 18 cm while cruising
    assert out.braking and out.v == 0.0 and out.state == DriveState.STOPPED
    assert "SAFETY STOP" in out.reason and "18 cm" in out.reason


def test_passthrough_when_clear():
    out = apply_safety_stop(CRUISE, 0.80, CFG)   # object far -> keep cruising
    assert out is CRUISE and out.v == 0.22


def test_passthrough_when_no_reading():
    assert apply_safety_stop(CRUISE, None, CFG) is CRUISE


def test_cm_unit_endtoend():
    cfg = SafetyConfig(enabled=True, safety_stop_distance_m=0.30, unit="cm")
    # reader would normalize first; simulate: 25 cm -> 0.25 m -> trips
    assert tof_triggered(normalize_range(25, cfg.unit), cfg)
    assert not tof_triggered(normalize_range(45, cfg.unit), cfg)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")

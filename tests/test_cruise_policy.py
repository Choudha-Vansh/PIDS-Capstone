"""Tests for the cruise/brake/resume policy and the hazard gate (pure python)."""
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from control.cruise_policy import CruisePolicy, CruiseConfig, DriveState
from control.hazard import HazardConfig, evaluate


# ── cruise policy ─────────────────────────────────────────────────────────────
def test_cruises_when_clear():
    p = CruisePolicy(CruiseConfig(cruise_speed_mps=0.2, brake_on_frames=3,
                                  resume_clear_frames=4))
    for _ in range(10):
        cmd = p.step(hazard=False)
    assert cmd.state == DriveState.CRUISE and cmd.v == 0.2 and not cmd.braking


def test_brakes_after_debounce():
    p = CruisePolicy(CruiseConfig(brake_on_frames=3, resume_clear_frames=4))
    assert not p.step(True).braking     # 1 hazard frame — not yet
    assert not p.step(True).braking     # 2 — not yet
    cmd = p.step(True)                  # 3 — brake
    assert cmd.braking and cmd.v == 0.0 and cmd.state == DriveState.STOPPED


def test_single_hazard_frame_does_not_brake():
    p = CruisePolicy(CruiseConfig(brake_on_frames=3))
    p.step(True)                        # one spurious hazard
    cmd = p.step(False)                 # cleared
    assert not cmd.braking and cmd.state == DriveState.CRUISE


def test_resume_only_after_clear_hysteresis():
    p = CruisePolicy(CruiseConfig(brake_on_frames=1, resume_clear_frames=4,
                                  cruise_speed_mps=0.2))
    assert p.step(True).braking         # stops immediately (brake_on_frames=1)
    for _ in range(3):                  # 3 clear frames — still holding stop
        c = p.step(False)
        assert c.v == 0.0 and c.state == DriveState.STOPPED
    c = p.step(False)                   # 4th clear frame — resume/accelerate
    assert c.v == 0.2 and c.state == DriveState.CRUISE and "ACCELERATE" in c.reason


# ── hazard gate ──────────────────────────────────────────────────────────────
@dataclass
class FakeDet:
    track_id: int
    bbox_xyxy: tuple
    @property
    def bbox_height_px(self):
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]
    @property
    def center_px(self):
        return ((self.bbox_xyxy[0]+self.bbox_xyxy[2])/2,
                (self.bbox_xyxy[1]+self.bbox_xyxy[3])/2)


def _cfg(**kw):
    base = dict(corridor_margin_m=0.3, min_bbox_height_frac=0.25,
                trigger_distance_m=2.0, focal_px=1000.0, assumed_ped_height_m=1.7)
    base.update(kw)
    return HazardConfig(**base)


def test_person_in_path_is_hazard():
    IMG_W, IMG_H = 640, 480
    # centered, tall box -> close and in path. height 300px: dist=1.7*1000/300=5.67m? too far.
    # use focal so distance <= 2m: height=1000 -> but >img_h. pick focal small.
    cfg = _cfg(focal_px=300.0, trigger_distance_m=3.0)  # dist=1.7*300/170=3.0
    det = FakeDet(1, (235, 155, 405, 325))              # 170px tall, centered
    r = evaluate([det], IMG_W, IMG_H, cfg)
    assert r.hazard and r.track_id == 1


def test_offpath_person_not_hazard():
    IMG_W, IMG_H = 640, 480
    cfg = _cfg(focal_px=300.0, trigger_distance_m=3.0)
    det = FakeDet(1, (5, 155, 175, 325))               # far left -> big lateral offset
    r = evaluate([det], IMG_W, IMG_H, cfg)
    assert not r.hazard


def test_too_small_person_ignored():
    IMG_W, IMG_H = 640, 480
    cfg = _cfg(min_bbox_height_frac=0.25)
    det = FakeDet(1, (310, 230, 330, 260))             # 30px tall < 0.25*480=120
    r = evaluate([det], IMG_W, IMG_H, cfg)
    assert not r.hazard


def test_intent_gate_blocks_when_not_crossing():
    IMG_W, IMG_H = 640, 480
    cfg = _cfg(focal_px=300.0, trigger_distance_m=3.0,
               use_intent_model=True, intent_prob_gate=0.5)
    det = FakeDet(1, (235, 155, 405, 325))
    assert not evaluate([det], IMG_W, IMG_H, cfg, intent_probs={1: 0.1}).hazard
    assert evaluate([det], IMG_W, IMG_H, cfg, intent_probs={1: 0.9}).hazard


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

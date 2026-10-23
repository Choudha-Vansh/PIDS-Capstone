"""
Prepare PIE annotations -> per-track sequences using the shared feature builder.

PIE's advantage over JAAD: it ships ego-vehicle OBD (speed) and richer intention
annotations, so we can populate the ego_speed feature (and, if available, use the
provided intention labels). PIE stores annotations as XML per set/video.

This uses the official PIE annotation layout:
    <pie_root>/annotations/set0X/video_XXXX_annt.xml
    <pie_root>/annotations_attributes/... (intention)
    <pie_root>/annotations_vehicle/set0X/video_XXXX_obd.xml (ego speed)

Because PIE layouts vary a little by release, the loader is defensive: if ego speed
isn't found it falls back to 0.0 and warns, rather than crashing.

Output: artifacts/pie_sequences.npz  with X (N,T,F), y (N,), clip_ids (N,)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from intent.features import FeatureConfig, build_frame_feature, slice_sequences  # noqa: E402

IMG_W, IMG_H = 1920.0, 1080.0


def _load_obd_speed(pie_root, set_name, video_name):
    """Return dict frame -> ego speed (m/s), or {} if not found."""
    patterns = [
        os.path.join(pie_root, "annotations_vehicle", set_name, f"{video_name}_obd.xml"),
        os.path.join(pie_root, "annotations_vehicle", set_name, f"{video_name}_vehicle.xml"),
    ]
    for p in patterns:
        if os.path.exists(p):
            speeds = {}
            root = ET.parse(p).getroot()
            for fr in root.iter():
                if fr.tag.lower() in ("frame", "obd"):
                    try:
                        fnum = int(fr.attrib.get("id") or fr.attrib.get("frame"))
                        # PIE OBD speed is usually km/h -> m/s
                        spd = float(fr.attrib.get("OBD_speed")
                                    or fr.attrib.get("speed") or 0.0)
                        speeds[fnum] = spd / 3.6
                    except (TypeError, ValueError):
                        continue
            return speeds
    return {}


def _parse_video(xml_path, speeds, fc, seq_len, hor):
    root = ET.parse(xml_path).getroot()
    outX, outY = [], []
    for tr in root.findall(".//track"):
        if tr.attrib.get("label") not in ("pedestrian", "ped"):
            continue
        rows = []
        for box in tr.findall("box"):
            if box.attrib.get("outside") == "1" or box.attrib.get("occluded") == "2":
                continue
            try:
                xtl = float(box.attrib["xtl"]); ytl = float(box.attrib["ytl"])
                xbr = float(box.attrib["xbr"]); ybr = float(box.attrib["ybr"])
            except (KeyError, ValueError):
                continue
            frame = int(box.attrib.get("frame", len(rows)))
            intent = 0
            for a in box.findall("attribute"):
                nm = a.attrib.get("name", "").lower()
                if nm in ("cross", "crossing") and (a.text or "").lower() == "crossing":
                    intent = 1
                if nm == "intention_prob":
                    try:
                        intent = 1 if float(a.text) >= 0.5 else 0
                    except (TypeError, ValueError):
                        pass
            rows.append([frame, xtl, ytl, xbr, ybr, intent])
        if len(rows) <= 4:
            continue
        rows = np.array(rows, np.float32)
        frames = rows[:, 0].astype(int)
        feats = []
        for r in rows:
            _, xtl, ytl, xbr, ybr, _ = r
            w = (xbr - xtl) / IMG_W; h = (ybr - ytl) / IMG_H
            xc = (xtl + xbr) / 2 / IMG_W; yc = (ytl + ybr) / 2 / IMG_H
            ego = speeds.get(int(r[0]), 0.0)
            bbox = np.array([xc, yc, w, h], np.float32)
            feats.append(build_frame_feature(bbox, None, None, None, 0.0, ego, fc))
        feats = np.stack(feats)
        X, y = slice_sequences(feats, rows[:, 5], frames, seq_len, hor)
        outX.extend(X); outY.extend(y)
    return outX, outY


def build(cfg):
    root = cfg["paths"]["pie_root"]
    ann_dir = os.path.join(root, "annotations")
    fc = FeatureConfig(use_pose=cfg["intent"]["use_pose"],
                       use_3d=cfg["intent"]["use_3d"])
    seq_len = cfg["intent"]["seq_length"]; hor = cfg["intent"]["predict_horizon"]

    xmls = glob.glob(os.path.join(ann_dir, "**", "*.xml"), recursive=True)
    if not xmls:
        print(f"WARNING: no PIE annotation XMLs under {ann_dir}. "
              f"Check paths.pie_root in config and SETUP.md.")
        return
    print(f"PIE annotation files: {len(xmls)}")

    allX, allY, clip_ids = [], [], []
    n_no_speed = 0
    for xml_path in xmls:
        video_name = os.path.splitext(os.path.basename(xml_path))[0].replace("_annt", "")
        set_name = os.path.basename(os.path.dirname(xml_path))
        speeds = _load_obd_speed(root, set_name, video_name)
        if not speeds:
            n_no_speed += 1
        X, y = _parse_video(xml_path, speeds, fc, seq_len, hor)
        allX.extend(X); allY.extend(y); clip_ids.extend([video_name] * len(X))

    if n_no_speed:
        print(f"NOTE: {n_no_speed} videos had no OBD speed found (ego_speed=0 there).")
    X = np.array(allX, np.float32); y = np.array(allY, np.float32)
    print(f"PIE sequences: {X.shape}  crossing={int(y.sum())}")
    os.makedirs(cfg["paths"]["artifacts"], exist_ok=True)
    out = os.path.join(cfg["paths"]["artifacts"], "pie_sequences.npz")
    np.savez_compressed(out, X=X, y=y, clip_ids=np.array(clip_ids))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    a = ap.parse_args()
    build(yaml.safe_load(open(a.config)))

"""
Prepare JAAD annotations -> per-track sequences with the shared feature builder.

JAAD is camera-only (no ego speed, no 3D). We fill 3D/ego features with zeros and
rely on pose + bbox kinematics. Carries the fixes from the earlier prototype:
  - skip outside=1 / fully-occluded boxes
  - gap-aware velocity (never differentiate across dropped frames)
  - keep real frame numbers so slice_sequences can skip windows spanning a gap

Output: artifacts/jaad_sequences.npz  with X (N,T,F), y (N,), clip_ids (N,)

JAAD's XML has no pose keypoints. When intent.use_pose is on, this runs YOLO-pose
(perception.detector.PoseDetector) over each clip's frames and matches detections
back to the annotated pedestrian boxes by IoU, so training sees real keypoints
instead of zeros -- matching what the live/demo path feeds the model. Missing clips
or unmatched boxes fall back to zero keypoints for that frame.
"""
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from intent.features import (FeatureConfig, build_frame_feature,  # noqa: E402
                             normalize_keypoints, slice_sequences)

IMG_W, IMG_H = 1920.0, 1080.0


def parse_tracks_for_file(xml_path):
    """One JAAD annotation XML -> {track_id: rows[frame, xtl, ytl, xbr, ybr, cross]}."""
    root = ET.parse(xml_path).getroot()
    tracks = {}
    for tr in root.findall(".//track"):
        if tr.attrib.get("label") != "pedestrian":
            continue
        tid = tr.attrib.get("id", str(id(tr)))
        rows = []
        for box in tr.findall("box"):
            if box.attrib.get("outside") == "1":
                continue
            if box.attrib.get("occluded") == "2":
                continue
            try:
                xtl = float(box.attrib["xtl"]); ytl = float(box.attrib["ytl"])
                xbr = float(box.attrib["xbr"]); ybr = float(box.attrib["ybr"])
            except (KeyError, ValueError):
                continue
            intent = 0
            for a in box.findall("attribute"):
                if a.attrib.get("name") == "cross" and a.text == "crossing":
                    intent = 1
            frame = int(box.attrib.get("frame", len(rows)))
            rows.append([frame, xtl, ytl, xbr, ybr, intent])
        if len(rows) > 4:
            tracks[tid] = np.array(rows, dtype=np.float32)
    return tracks


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def build_pose_cache(video_path, needed_frames, detector):
    """Run YOLO-pose over just the annotated frames of one clip.

    Sequential grab()-only skip for frames we don't need keeps this from
    decoding every frame in the clip, just the ones with an annotated box.
    """
    import cv2
    cache = {}
    cap = cv2.VideoCapture(video_path)
    idx = 0
    max_frame = max(needed_frames) if needed_frames else -1
    while idx <= max_frame:
        if idx in needed_frames:
            ok, frame = cap.read()
            if not ok:
                break
            cache[idx] = detector.detect(frame)
        else:
            ok = cap.grab()
            if not ok:
                break
        idx += 1
    cap.release()
    return cache


def match_keypoints(bbox_xyxy, dets, iou_thresh=0.3):
    best, best_iou = None, iou_thresh
    for d in dets:
        i = _iou(bbox_xyxy, d.bbox_xyxy)
        if i > best_iou:
            best_iou, best = i, d
    return best.keypoints if best is not None else None


def build(cfg):
    ann = os.path.join(cfg["paths"]["jaad_root"], "annotations")
    if not os.path.isdir(ann):
        # some JAAD layouts keep XML at the root
        ann = cfg["paths"]["jaad_root"]
    clips_dir = os.path.join(cfg["paths"]["jaad_root"], "JAAD_clips")
    fc = FeatureConfig(use_pose=cfg["intent"]["use_pose"],
                       use_3d=cfg["intent"]["use_3d"])
    seq_len = cfg["intent"]["seq_length"]; hor = cfg["intent"]["predict_horizon"]

    detector = None
    if fc.use_pose:
        from perception.detector import PoseDetector
        detector = PoseDetector(cfg["perception"]["pose_model"],
                                conf=cfg["perception"]["det_conf"])

    out = os.path.join(cfg["paths"]["artifacts"], "jaad_sequences.npz")
    ckpt = os.path.join(cfg["paths"]["artifacts"], "jaad_sequences.partial.npz")
    os.makedirs(cfg["paths"]["artifacts"], exist_ok=True)

    allX, allY, clip_ids = [], [], []
    done_files = set()
    if os.path.exists(ckpt):
        d = np.load(ckpt, allow_pickle=True)
        allX = list(d["X"]); allY = list(d["y"]); clip_ids = list(d["clip_ids"])
        done_files = set(d["done_files"].tolist())
        print(f"Resuming from checkpoint: {len(allX)} sequences, "
              f"{len(done_files)} videos already done")

    xml_files = sorted(f for f in os.listdir(ann) if f.endswith(".xml"))
    n_tracks = 0
    n_missing_clips = 0
    checkpoint_every = 10

    for fi, fname in enumerate(xml_files):
        if fname in done_files:
            continue
        video_id = os.path.splitext(fname)[0]
        tracks = parse_tracks_for_file(os.path.join(ann, fname))
        if not tracks:
            done_files.add(fname)
            continue
        n_tracks += len(tracks)

        pose_cache = {}
        if detector is not None:
            video_path = os.path.join(clips_dir, f"{video_id}.mp4")
            if os.path.isfile(video_path):
                needed = set()
                for rows in tracks.values():
                    needed.update(int(f) for f in rows[:, 0])
                pose_cache = build_pose_cache(video_path, needed, detector)
            else:
                n_missing_clips += 1

        for tid, rows in tracks.items():
            frames = rows[:, 0].astype(int)
            feats = []
            for r in rows:
                frame_idx, xtl, ytl, xbr, ybr, _ = r
                w = (xbr - xtl) / IMG_W; h = (ybr - ytl) / IMG_H
                xc = (xtl + xbr) / 2 / IMG_W; yc = (ytl + ybr) / 2 / IMG_H
                bbox = np.array([xc, yc, w, h], np.float32)
                kp_norm = None
                if detector is not None:
                    dets = pose_cache.get(int(frame_idx), [])
                    kp_px = match_keypoints((xtl, ytl, xbr, ybr), dets)
                    if kp_px is not None:
                        kp_norm = normalize_keypoints(kp_px, int(IMG_W), int(IMG_H))
                feats.append(build_frame_feature(bbox, kp_norm, None, None, 0.0, 0.0, fc))
            feats = np.stack(feats)
            labels = rows[:, 5]
            X, y = slice_sequences(feats, labels, frames, seq_len, hor)
            allX.extend(X); allY.extend(y); clip_ids.extend([f"{fname}_{tid}"] * len(X))

        done_files.add(fname)
        if (fi + 1) % checkpoint_every == 0:
            np.savez_compressed(
                ckpt, X=np.array(allX, np.float32), y=np.array(allY, np.float32),
                clip_ids=np.array(clip_ids), done_files=np.array(sorted(done_files)))
            print(f"  checkpoint: {fi + 1}/{len(xml_files)} videos, "
                  f"{len(allX)} sequences so far", flush=True)

    print(f"JAAD pedestrian tracks: {n_tracks}")
    if n_missing_clips:
        print(f"WARNING: {n_missing_clips} clips missing from {clips_dir}; "
              f"pose left zero for those videos' tracks")

    X = np.array(allX, np.float32); y = np.array(allY, np.float32)
    print(f"JAAD sequences: {X.shape}  crossing={int(y.sum())}")
    np.savez_compressed(out, X=X, y=y, clip_ids=np.array(clip_ids))
    print(f"Saved -> {out}")
    if os.path.exists(ckpt):
        os.remove(ckpt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    a = ap.parse_args()
    build(yaml.safe_load(open(a.config)))

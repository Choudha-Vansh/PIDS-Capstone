"""
Multimodal sensor source backed by nuScenes (camera + radar + LIDAR + ego speed).
Requires nuscenes-devkit and a downloaded nuScenes (start with v1.0-mini).

This yields real synchronized sensor data so perception/fusion.py can be trained
and validated against 3D ground-truth boxes. Import of nuscenes is lazy so the rest
of the project runs without the devkit installed.
"""
from __future__ import annotations

import os
from typing import Iterator, Optional

import numpy as np

from sensors.base import RadarReturn, SensorFrame, SensorSource


class NuScenesSource(SensorSource):
    def __init__(self, dataroot: str, version: str = "v1.0-mini",
                 scene_index: int = 0, camera: str = "CAM_FRONT",
                 radar: str = "RADAR_FRONT", lidar: str = "LIDAR_TOP"):
        try:
            from nuscenes.nuscenes import NuScenes  # lazy import
        except ImportError as e:
            raise ImportError(
                "nuscenes-devkit not installed. `pip install nuscenes-devkit` "
                "and download nuScenes (see SETUP.md)."
            ) from e

        import cv2  # local import keeps module import cheap
        self._cv2 = cv2
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = dataroot
        self.camera = camera
        self.radar = radar
        self.lidar = lidar
        self._scene = self.nusc.scene[scene_index]
        self._fps = 12.0  # nuScenes keyframes ~2 Hz; sample_data ~12 Hz for camera

    @property
    def fps(self) -> float:
        return self._fps

    def _load_radar(self, sample) -> list:
        from nuscenes.utils.data_classes import RadarPointCloud
        try:
            tok = sample["data"][self.radar]
        except KeyError:
            return []
        sd = self.nusc.get("sample_data", tok)
        pc = RadarPointCloud.from_file(os.path.join(self.dataroot, sd["filename"]))
        pts = pc.points  # 18 x N; rows 0,1=x,y ; 8,9 = vx,vy (compensated)
        returns = []
        for i in range(pts.shape[1]):
            x, y = float(pts[0, i]), float(pts[1, i])
            vx, vy = float(pts[8, i]), float(pts[9, i])
            rng = float(np.hypot(x, y))
            radial_v = float((x * vx + y * vy) / rng) if rng > 1e-6 else 0.0
            returns.append(RadarReturn(x, y, rng, radial_v))
        return returns

    def _load_lidar(self, sample) -> Optional[np.ndarray]:
        from nuscenes.utils.data_classes import LidarPointCloud
        try:
            tok = sample["data"][self.lidar]
        except KeyError:
            return None
        sd = self.nusc.get("sample_data", tok)
        pc = LidarPointCloud.from_file(os.path.join(self.dataroot, sd["filename"]))
        return pc.points[:3, :].T.astype(np.float32)  # Nx3

    def __iter__(self) -> Iterator[SensorFrame]:
        sample_tok = self._scene["first_sample_token"]
        idx = 0
        while sample_tok:
            sample = self.nusc.get("sample", sample_tok)
            cam_sd = self.nusc.get("sample_data", sample["data"][self.camera])
            img = self._cv2.imread(os.path.join(self.dataroot, cam_sd["filename"]))

            # ego speed from consecutive ego poses
            ego_speed = self._ego_speed(sample)

            yield SensorFrame(
                frame_idx=idx,
                timestamp_s=sample["timestamp"] / 1e6,
                image=img,
                ego_speed_mps=ego_speed,
                radar=self._load_radar(sample),
                lidar_points=self._load_lidar(sample),
                calib={"sample_token": sample_tok},
            )
            sample_tok = sample["next"]
            idx += 1

    def _ego_speed(self, sample) -> float:
        try:
            sd = self.nusc.get("sample_data", sample["data"][self.lidar])
            pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
            if not hasattr(self, "_prev_pose"):
                self._prev_pose = (pose["translation"], sample["timestamp"])
                return 0.0
            (px, py, _), pt = self._prev_pose
            dt = (sample["timestamp"] - pt) / 1e6
            cx, cy, _ = pose["translation"]
            self._prev_pose = (pose["translation"], sample["timestamp"])
            return float(np.hypot(cx - px, cy - py) / dt) if dt > 0 else 0.0
        except Exception:
            return 0.0

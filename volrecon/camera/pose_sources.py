"""Pose source strategies for ZED capture."""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import yaml

from volrecon.geometry.transforms import make_T


class PoseSource(ABC):
    @abstractmethod
    def get_T_world_left(self, frame_idx: int, timestamp_ns: int) -> tuple[np.ndarray | None, str]:
        """Return (T_world_left 4x4, tracking_state string)."""


class ZEDTrackingPoseSource(PoseSource):
    """Pose comes directly from ZEDStereoFrame (already fetched in grab_frame)."""

    def get_T_world_left(self, frame_idx: int, timestamp_ns: int) -> tuple[np.ndarray | None, str]:
        return None, "zed_tracking_deferred"


class FixedRigPoseSource(PoseSource):
    def __init__(self, rig_yaml: Path, serial_number: str) -> None:
        data = yaml.safe_load(rig_yaml.read_text(encoding="utf-8"))
        cameras = data.get("cameras", data)
        key = str(serial_number)
        if key not in cameras:
            raise KeyError(f"Serial {serial_number} not found in rig calibration {rig_yaml}")
        entry = cameras[key]
        T = np.asarray(entry["T_world_left"], dtype=np.float64).reshape(4, 4)
        self.T_world_left = T

    def get_T_world_left(self, frame_idx: int, timestamp_ns: int) -> tuple[np.ndarray | None, str]:
        return self.T_world_left.copy(), "fixed_rig"


class ExternalPoseSource(PoseSource):
    def __init__(self, pose_file: Path) -> None:
        self.by_frame: dict[int, np.ndarray] = {}
        self.by_ts: dict[int, np.ndarray] = {}
        if pose_file.suffix.lower() == ".json":
            data = json.loads(pose_file.read_text(encoding="utf-8"))
            for entry in data:
                T = np.asarray(entry["T_world_left"], dtype=np.float64).reshape(4, 4)
                if "frame_idx" in entry:
                    self.by_frame[int(entry["frame_idx"])] = T
                if "timestamp_ns" in entry:
                    self.by_ts[int(entry["timestamp_ns"])] = T
        elif pose_file.suffix.lower() == ".csv":
            with pose_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    T = np.eye(4)
                    if "T_world_left" in row:
                        T = np.asarray(json.loads(row["T_world_left"]), dtype=np.float64).reshape(4, 4)
                    elif all(k in row for k in ("tx", "ty", "tz")):
                        T = make_T(np.eye(3), [float(row["tx"]), float(row["ty"]), float(row["tz"])])
                    if "frame_idx" in row:
                        self.by_frame[int(row["frame_idx"])] = T
                    if "timestamp_ns" in row:
                        self.by_ts[int(row["timestamp_ns"])] = T
        else:
            raise ValueError(f"Unsupported external pose file: {pose_file}")

    def get_T_world_left(self, frame_idx: int, timestamp_ns: int) -> tuple[np.ndarray | None, str]:
        if frame_idx in self.by_frame:
            return self.by_frame[frame_idx].copy(), "external_pose"
        if timestamp_ns in self.by_ts:
            return self.by_ts[timestamp_ns].copy(), "external_pose"
        return None, "external_pose_missing"


def build_pose_source(
    mode: str,
    rig_yaml: Path | None = None,
    external_pose_file: Path | None = None,
    serial_number: str = "unknown",
) -> PoseSource | None:
    if mode == "zed_tracking":
        return ZEDTrackingPoseSource()
    if mode == "fixed_rig_yaml":
        if rig_yaml is None:
            raise ValueError("fixed_rig_yaml mode requires rig_calibration.yaml path")
        return FixedRigPoseSource(rig_yaml, serial_number)
    if mode == "external_poses":
        if external_pose_file is None:
            raise ValueError("external_poses mode requires external_pose_file")
        return ExternalPoseSource(external_pose_file)
    if mode == "none":
        return None
    raise ValueError(f"Unknown pose mode: {mode}")

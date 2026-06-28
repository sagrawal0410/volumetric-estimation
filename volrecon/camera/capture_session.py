"""Capture session with keyframe selection."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from volrecon.camera.pose_sources import PoseSource, build_pose_source
from volrecon.camera.zed_capture import ZEDCaptureConfig, ZEDStereoCapture, ZEDStereoFrame
from volrecon.camera.zed_manifest import write_scene_manifest, write_scene_meta
from volrecon.config import PROJECT_ROOT
from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.io.json_io import write_json

logger = logging.getLogger(__name__)


def _rotation_angle_deg(R: np.ndarray) -> float:
    trace = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(trace)))


@dataclass
class CaptureSessionConfig:
    scene_name: str
    output_root: Path = Path("data/zed_captures")
    num_keyframes: int = 30
    save_every_n_frames: int = 5
    fixed_camera: bool = False
    pose_mode: str = "zed_tracking"
    fixed_rig_yaml: Path | None = None
    external_pose_file: Path | None = None
    overwrite: bool = False
    save_preview_video: bool = False


class CaptureSession:
    def __init__(
        self,
        capture: ZEDStereoCapture,
        cfg: CaptureSessionConfig,
        zed_cfg: ZEDCaptureConfig,
    ) -> None:
        self.capture = capture
        self.cfg = cfg
        self.zed_cfg = zed_cfg
        self.scene_dir: Path | None = None
        self.views: list[ViewRecord] = []
        self._last_saved_T: np.ndarray | None = None
        self._last_saved_ts: float = 0.0
        self._saved_count = 0
        self._pose_source: PoseSource | None = None
        self._preview_writer = None

    def start_scene(self, scene_name: str | None = None) -> Path:
        name = scene_name or self.cfg.scene_name
        self.scene_dir = (self.cfg.output_root / name).resolve()
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        (self.scene_dir / "views").mkdir(exist_ok=True)
        (self.scene_dir / "runs").mkdir(exist_ok=True)

        self.capture.save_camera_info(self.scene_dir)
        write_scene_meta(self.scene_dir, name, {"pose_mode": self.cfg.pose_mode})
        write_json(self.scene_dir / "capture_config.json", {
            "zed": self.zed_cfg.__dict__,
            "session": {k: str(v) if isinstance(v, Path) else v for k, v in self.cfg.__dict__.items()},
        })

        calib = self.capture.get_calibration()
        self._pose_source = build_pose_source(
            self.cfg.pose_mode,
            self.cfg.fixed_rig_yaml,
            self.cfg.external_pose_file,
            str(calib.get("camera_serial", "unknown")),
        )

        if self.cfg.save_preview_video:
            w, h = calib["image_width"], calib["image_height"]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._preview_writer = cv2.VideoWriter(
                str(self.scene_dir / "preview.mp4"),
                fourcc,
                self.zed_cfg.camera_fps,
                (w * 2, h),
            )
        return self.scene_dir

    def should_save_keyframe(self, frame: ZEDStereoFrame) -> bool:
        if self.cfg.fixed_camera:
            return frame.frame_idx % max(self.cfg.save_every_n_frames, 1) == 0

        if self._last_saved_T is None:
            return True
        if frame.T_world_left is None:
            elapsed = time.time() - self._last_saved_ts
            return elapsed >= self.zed_cfg.min_seconds_between_keyframes

        dt = np.linalg.norm(frame.T_world_left[:3, 3] - self._last_saved_T[:3, 3])
        R = frame.T_world_left[:3, :3] @ self._last_saved_T[:3, :3].T
        ang = _rotation_angle_deg(R)
        elapsed = time.time() - self._last_saved_ts
        return (
            dt >= self.zed_cfg.min_translation_between_keyframes_m
            or ang >= self.zed_cfg.min_rotation_between_keyframes_deg
            or elapsed >= self.zed_cfg.min_seconds_between_keyframes
        )

    def save_keyframe(self, frame: ZEDStereoFrame) -> ViewRecord:
        if self.scene_dir is None:
            raise RuntimeError("Call start_scene() first")
        view_id = f"{self._saved_count:06d}"
        view_dir = self.scene_dir / "views" / view_id
        if view_dir.exists() and not self.cfg.overwrite:
            logger.info("Skip existing keyframe %s", view_id)
            from volrecon.io.json_io import read_jsonl
            from volrecon.datasets.canonical_schema import ViewRecord

            manifest = self.scene_dir / "manifest.jsonl"
            if manifest.exists():
                for row in read_jsonl(manifest):
                    if row.get("view_id") == view_id:
                        return ViewRecord.from_dict(row)
            meta_path = view_dir / "meta.json"
            if meta_path.exists():
                import json

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                calib = {
                    "K_left": meta["K_left"],
                    "K_right": meta["K_right"],
                    "T_left_right": meta["T_left_right"],
                    "baseline_m": meta["baseline_m"],
                    "image_width": meta["image_width"],
                    "image_height": meta["image_height"],
                }
                from volrecon.camera.zed_manifest import frame_to_view_record

                return frame_to_view_record(
                    self.scene_dir.name,
                    view_id,
                    {"left": meta["left_path"], "right": meta["right_path"]},
                    calib,
                    meta,
                )
            self._saved_count += 1
            return ViewRecord(
                dataset="zed_live",
                scene_id=self.scene_dir.name,
                view_id=view_id,
            )

        if self._pose_source and self.cfg.pose_mode != "zed_tracking":
            T, state = self._pose_source.get_T_world_left(frame.frame_idx, frame.timestamp_ns)
            if T is not None:
                frame.T_world_left = T
                frame.tracking_state = state

        paths = self.capture.save_frame(frame, self.scene_dir, view_id)
        rec = self.capture.build_view_record(frame, paths, self.scene_dir.name, view_id)
        self.views.append(rec)
        self._saved_count += 1

        if self._preview_writer is not None:
            grid = np.hstack([cv2.cvtColor(frame.left_rgb, cv2.COLOR_RGB2BGR),
                              cv2.cvtColor(frame.right_rgb, cv2.COLOR_RGB2BGR)])
            self._preview_writer.write(grid)

        if frame.T_world_left is not None:
            self._last_saved_T = frame.T_world_left.copy()
        self._last_saved_ts = time.time()
        return rec

    def finish_scene(self) -> Path:
        if self.scene_dir is None:
            raise RuntimeError("No active scene")
        write_scene_manifest(self.scene_dir, self.views, PROJECT_ROOT)
        meta_path = self.scene_dir / "scene_meta.json"
        data = {"scene_id": self.scene_dir.name, "num_views": len(self.views), "dataset": "zed_live"}
        write_json(meta_path, data)
        if self._preview_writer is not None:
            self._preview_writer.release()
        logger.info("Scene saved: %s (%d views)", self.scene_dir, len(self.views))
        return self.scene_dir

    def capture_keyframes(self, num_keyframes: int | None = None) -> Path:
        n = num_keyframes or self.cfg.num_keyframes
        self.start_scene()
        while self._saved_count < n:
            frame = self.capture.grab_frame()
            if frame is None:
                break
            if self.should_save_keyframe(frame):
                self.save_keyframe(frame)
        return self.finish_scene()

"""ZED 2i stereo RGB capture (no SDK depth)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from volrecon.camera.camera_health import ZEDDepthCallGuard
from volrecon.camera.zed_backend import get_sl_module
from volrecon.camera.zed_calibration import extract_zed_calibration
from volrecon.camera.zed_manifest import build_view_meta, frame_to_view_record
from volrecon.config import PROJECT_ROOT
from volrecon.geometry.transforms import make_T
from volrecon.io.image_io import write_image
from volrecon.io.json_io import write_json

logger = logging.getLogger(__name__)

RESOLUTION_MAP = {
    "HD2K": "HD2K",
    "HD1080": "HD1080",
    "HD720": "HD720",
    "VGA": "VGA",
}


@dataclass
class ZEDCaptureConfig:
    camera_resolution: str = "HD720"
    camera_fps: int = 15
    coordinate_units: str = "METER"
    depth_mode: str = "NONE"
    enable_positional_tracking: bool = False
    positional_tracking_area_file: str | None = None
    min_seconds_between_keyframes: float = 0.25
    min_translation_between_keyframes_m: float = 0.03
    min_rotation_between_keyframes_deg: float = 5.0
    max_frames: int | None = None
    output_resolution_scale: float = 1.0
    save_preview_video: bool = False
    exposure: int | None = None
    gain: int | None = None
    white_balance: int | None = None
    svo_input_path: str | None = None
    svo_realtime: bool = True
    serial_number: int | None = None
    mock_camera: bool = False


@dataclass
class ZEDStereoFrame:
    frame_idx: int
    timestamp_ns: int
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    K_left: np.ndarray
    K_right: np.ndarray
    T_left_right: np.ndarray
    baseline_m: float
    image_width: int
    image_height: int
    T_world_left: np.ndarray | None = None
    tracking_state: str | None = None
    camera_serial: str = "unknown"
    sdk_version: str = "unknown"


def bgra_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class ZEDStereoCapture:
    def __init__(self, config: ZEDCaptureConfig) -> None:
        self.config = config
        self.sl = get_sl_module()
        self._zed_raw = self.sl.Camera()
        self.zed = ZEDDepthCallGuard(self._zed_raw)
        self._left_mat = self.sl.Mat()
        self._right_mat = self.sl.Mat()
        self._calib: dict[str, Any] | None = None
        self._frame_idx = 0
        self._opened = False
        self._sdk_version = "unknown"

    def open(self) -> None:
        init = self.sl.InitParameters()
        res_name = self.config.camera_resolution.upper()
        if hasattr(self.sl, "RESOLUTION_MAP"):
            init.camera_resolution = self.sl.RESOLUTION_MAP.get(res_name, self.sl.RESOLUTION.HD720)
        elif hasattr(self.sl, "RESOLUTION"):
            init.camera_resolution = getattr(self.sl.RESOLUTION, res_name, self.sl.RESOLUTION.HD720)
        init.camera_fps = self.config.camera_fps
        init.coordinate_units = self.sl.UNIT.METER

        if hasattr(self.sl, "DEPTH_MODE"):
            init.depth_mode = self.sl.DEPTH_MODE.NONE
            logger.info("ZED depth_mode set to NONE — SDK depth will NOT be used for inference.")

        if self.config.svo_input_path:
            if hasattr(self.sl, "InputType"):
                init.input = self.sl.InputType()
                init.input.set_from_svo_file(self.config.svo_input_path)
            else:
                init.input = self.config.svo_input_path
            init.svo_real_time_mode = self.config.svo_realtime

        if self.config.serial_number is not None and hasattr(init, "set_from_serial_number"):
            init.set_from_serial_number = self.config.serial_number

        status = self.zed.open(init)
        success = status == self.sl.ERROR_CODE.SUCCESS if hasattr(self.sl, "ERROR_CODE") else status == 0
        if not success:
            raise RuntimeError(f"Failed to open ZED camera/SVO: {status}")

        self._opened = True
        self._sdk_version = self.sl.get_sdk_version() if hasattr(self.sl, "get_sdk_version") else "unknown"

        if self.config.enable_positional_tracking:
            pt_params = self.sl.PositionalTrackingParameters()
            if self.config.positional_tracking_area_file:
                pt_params.area_file = self.config.positional_tracking_area_file
            pt_status = self.zed.enable_positional_tracking(pt_params)
            logger.info(
                "Positional tracking enabled (pose metadata only, NOT depth input). status=%s",
                pt_status,
            )

        self._calib = self.get_calibration()
        logger.info(
            "ZED opened serial=%s baseline=%.4fm resolution=%dx%d sdk=%s",
            self._calib.get("camera_serial"),
            self._calib["baseline_m"],
            self._calib["image_width"],
            self._calib["image_height"],
            self._sdk_version,
        )

    def close(self) -> None:
        if self._opened:
            self.zed.close()
            self._opened = False

    def get_calibration(self, output_resolution: tuple[int, int] | None = None) -> dict[str, Any]:
        calib = extract_zed_calibration(self.zed, output_resolution)
        if self.config.output_resolution_scale != 1.0 and output_resolution is None:
            w = int(calib["image_width"] * self.config.output_resolution_scale)
            h = int(calib["image_height"] * self.config.output_resolution_scale)
            calib = extract_zed_calibration(self.zed, (w, h))
        self._calib = calib
        return calib

    def _zed_pose_to_T(self, pose) -> np.ndarray:
        pd = pose.pose_data
        R = np.asarray(pd.rotation, dtype=np.float64).reshape(3, 3)
        t = np.asarray(pd.translation, dtype=np.float64).reshape(3)
        return make_T(R, t)

    def grab_frame(self) -> Optional[ZEDStereoFrame]:
        if not self._opened:
            raise RuntimeError("ZED camera not opened")
        if self.config.max_frames is not None and self._frame_idx >= self.config.max_frames:
            return None

        runtime = self.sl.RuntimeParameters()
        grab_status = self.zed.grab(runtime)
        ok = grab_status == self.sl.ERROR_CODE.SUCCESS if hasattr(self.sl, "ERROR_CODE") else grab_status == 0
        if not ok:
            return None

        calib = self._calib or self.get_calibration()
        w, h = calib["image_width"], calib["image_height"]

        self.zed.retrieve_image(self._left_mat, self.sl.VIEW.LEFT, self.sl.MEM.CPU)
        self.zed.retrieve_image(self._right_mat, self.sl.VIEW.RIGHT, self.sl.MEM.CPU)
        left_rgb = bgra_to_rgb(self._left_mat.get_data())
        right_rgb = bgra_to_rgb(self._right_mat.get_data())

        if left_rgb.shape[1] != w or left_rgb.shape[0] != h:
            left_rgb = cv2.resize(left_rgb, (w, h), interpolation=cv2.INTER_AREA)
            right_rgb = cv2.resize(right_rgb, (w, h), interpolation=cv2.INTER_AREA)

        ts = self._frame_idx * 33_000_000
        if hasattr(self.zed, "get_timestamp"):
            ts_obj = self.zed.get_timestamp(getattr(self.sl, "TIME_REFERENCE", None))
            if hasattr(ts_obj, "get_nanoseconds"):
                ts = int(ts_obj.get_nanoseconds())

        T_world_left = None
        tracking_state = "disabled"
        if self.config.enable_positional_tracking:
            pose = self.sl.Pose()
            pstatus = self.zed.get_position(pose, self.sl.REFERENCE_FRAME.WORLD)
            ps_ok = pstatus == self.sl.ERROR_CODE.SUCCESS if hasattr(self.sl, "ERROR_CODE") else pstatus == 0
            if ps_ok:
                T_world_left = self._zed_pose_to_T(pose)
                tracking_state = "ok"
            else:
                tracking_state = "unreliable"

        frame = ZEDStereoFrame(
            frame_idx=self._frame_idx,
            timestamp_ns=int(ts),
            left_rgb=left_rgb,
            right_rgb=right_rgb,
            K_left=np.asarray(calib["K_left"]),
            K_right=np.asarray(calib["K_right"]),
            T_left_right=np.asarray(calib["T_left_right"]),
            baseline_m=float(calib["baseline_m"]),
            image_width=w,
            image_height=h,
            T_world_left=T_world_left,
            tracking_state=tracking_state,
            camera_serial=str(calib.get("camera_serial", "unknown")),
            sdk_version=self._sdk_version,
        )
        self._frame_idx += 1
        return frame

    def save_frame(self, frame: ZEDStereoFrame, scene_dir: Path, view_id: str) -> dict[str, Path]:
        view_dir = scene_dir / "views" / view_id
        view_dir.mkdir(parents=True, exist_ok=True)
        left_path = view_dir / "left.png"
        right_path = view_dir / "right.png"
        write_image(left_path, frame.left_rgb)
        write_image(right_path, frame.right_rgb)
        return {"left": left_path, "right": right_path}

    def build_view_record(self, frame: ZEDStereoFrame, paths: dict[str, Path], scene_id: str, view_id: str):
        rel = {k: str(v.relative_to(PROJECT_ROOT) if v.is_absolute() else v) for k, v in paths.items()}
        calib = {
            "K_left": frame.K_left,
            "K_right": frame.K_right,
            "T_left_right": frame.T_left_right,
            "baseline_m": frame.baseline_m,
            "image_width": frame.image_width,
            "image_height": frame.image_height,
        }
        meta = build_view_meta(
            frame.frame_idx,
            frame.timestamp_ns,
            {"left": rel["left"], "right": rel["right"]},
            calib,
            frame.T_world_left,
            frame.tracking_state,
            frame.camera_serial,
            frame.sdk_version,
        )
        view_dir = paths["left"].parent
        write_json(view_dir / "meta.json", meta)
        return frame_to_view_record(scene_id, view_id, {"left": rel["left"], "right": rel["right"]}, calib, meta)

    def save_camera_info(self, scene_dir: Path) -> None:
        calib = self._calib or self.get_calibration()
        info = {
            "sdk_version": self._sdk_version,
            "camera_serial": calib.get("camera_serial"),
            "resolution": [calib["image_width"], calib["image_height"]],
            "fps": self.config.camera_fps,
            "baseline_m": calib["baseline_m"],
            "depth_mode": "NONE",
            "notes": "Inference uses rectified RGB only; no ZED SDK depth.",
        }
        write_json(scene_dir / "camera_info.json", info)
        write_json(scene_dir / "calibration.json", {
            "K_left": calib["K_left"].tolist(),
            "K_right": calib["K_right"].tolist(),
            "T_left_right": calib["T_left_right"].tolist(),
            "baseline_m": calib["baseline_m"],
        })
        np.save(scene_dir / "K_left.npy", calib["K_left"])
        np.save(scene_dir / "K_right.npy", calib["K_right"])
        np.save(scene_dir / "T_left_right.npy", calib["T_left_right"])

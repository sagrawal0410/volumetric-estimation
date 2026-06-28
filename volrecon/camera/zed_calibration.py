"""Extract ZED camera calibration into canonical numpy/JSON form."""

from __future__ import annotations

from typing import Any

import numpy as np

from volrecon.geometry.camera import resize_intrinsics
from volrecon.geometry.transforms import make_T


def _get_calib_params(camera_info) -> Any:
    if hasattr(camera_info, "camera_configuration"):
        return camera_info.camera_configuration.calibration_parameters
    if hasattr(camera_info, "calibration_parameters"):
        return camera_info.calibration_parameters
    raise AttributeError("Could not find calibration_parameters on ZED camera_info")


def _cam_to_K(cam, width: int, height: int) -> tuple[np.ndarray, list[float] | None]:
    if not hasattr(cam, "fx"):
        raise AttributeError("ZED camera params missing fx/fy/cx/cy")
    K = np.array(
        [[float(cam.fx), 0.0, float(cam.cx)], [0.0, float(cam.fy), float(cam.cy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = None
    if hasattr(cam, "disto"):
        dist = [float(x) for x in cam.disto]
    return K, dist


def _stereo_transform_to_T(stereo_transform) -> np.ndarray:
    if hasattr(stereo_transform, "m"):
        M = np.array(stereo_transform.m, dtype=np.float64).reshape(4, 4)
        return M
    R = np.eye(3)
    t = np.zeros(3)
    if hasattr(stereo_transform, "get_rotation_matrix"):
        rot = stereo_transform.get_rotation_matrix()
        if hasattr(rot, "r"):
            R = np.array(rot.r, dtype=np.float64).reshape(3, 3)
    if hasattr(stereo_transform, "get_translation"):
        tr = stereo_transform.get_translation()
        if hasattr(tr, "get"):
            t = np.array(tr.get(), dtype=np.float64).reshape(3)
    return make_T(R, t)


def extract_zed_calibration(
    zed,
    output_resolution: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """
    Extract left/right intrinsics, baseline, and T_left_right from ZED SDK.

    Raises if baseline cannot be determined.
    """
    info = zed.get_camera_information()
    calib = _get_calib_params(info)

    if hasattr(info, "camera_configuration") and hasattr(info.camera_configuration, "resolution"):
        res = info.camera_configuration.resolution
        src_w, src_h = int(res.width), int(res.height)
    else:
        src_w, src_h = 1280, 720

    if not hasattr(calib, "left_cam") or not hasattr(calib, "right_cam"):
        raise AttributeError("ZED calibration missing left_cam/right_cam")

    K_left, dist_l = _cam_to_K(calib.left_cam, src_w, src_h)
    K_right, dist_r = _cam_to_K(calib.right_cam, src_w, src_h)

    baseline_m = None
    T_left_right = np.eye(4, dtype=np.float64)

    if hasattr(calib, "stereo_transform"):
        T_left_right = _stereo_transform_to_T(calib.stereo_transform)
        baseline_m = float(abs(T_left_right[0, 3]))

    if baseline_m is None or baseline_m <= 0:
        if hasattr(calib, "baseline"):
            baseline_m = float(calib.baseline)
            T_left_right = make_T(np.eye(3), np.array([baseline_m, 0.0, 0.0]))

    if baseline_m is None or baseline_m <= 0:
        raise ValueError(
            "Could not determine ZED stereo baseline. FoundationStereo requires baseline for metric depth."
        )

    out_w, out_h = src_w, src_h
    if output_resolution is not None:
        out_w, out_h = output_resolution
        sx, sy = out_w / src_w, out_h / src_h
        K_left = resize_intrinsics(K_left, sx, sy)
        K_right = resize_intrinsics(K_right, sx, sy)

    serial = "unknown"
    if hasattr(info, "serial_number"):
        serial = str(info.serial_number)

    return {
        "K_left": K_left,
        "K_right": K_right,
        "T_left_right": T_left_right,
        "baseline_m": baseline_m,
        "distortion_left": dist_l,
        "distortion_right": dist_r,
        "image_width": out_w,
        "image_height": out_h,
        "source_width": src_w,
        "source_height": src_h,
        "camera_serial": serial,
    }

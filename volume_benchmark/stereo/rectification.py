"""Stereo rectification and validation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def validate_rectified_pair(
    left: np.ndarray,
    right: np.ndarray,
    optional_correspondence_debug: bool = False,
    bypass_rectification: bool = False,
) -> dict[str, Any]:
    """
    Check that left/right are same shape and optionally estimate vertical disparity.

    Set bypass_rectification=True for pre-rectified pairs (e.g. ZED SDK left/right).
    """
    if left.shape != right.shape:
        raise ValueError(f"Left/right shape mismatch: {left.shape} vs {right.shape}")
    if left.ndim != 3 or left.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {left.shape}")

    report: dict[str, Any] = {
        "height": int(left.shape[0]),
        "width": int(left.shape[1]),
        "channels": int(left.shape[2]),
        "rectified_assumed": True,
        "bypass_rectification": bypass_rectification,
    }

    if bypass_rectification:
        return report

    if optional_correspondence_debug:
        gray_l = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(500)
        kp1, des1 = orb.detectAndCompute(gray_l, None)
        kp2, des2 = orb.detectAndCompute(gray_r, None)
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if matches:
                dy = [abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) for m in matches]
                report["median_vertical_disparity_px"] = float(np.median(dy))
                report["num_matches"] = len(matches)
    return report


def rectify_pair_opencv(
    left: np.ndarray,
    right: np.ndarray,
    K_left: np.ndarray,
    D_left: np.ndarray,
    K_right: np.ndarray,
    D_right: np.ndarray,
    R_left_to_right: np.ndarray,
    t_left_to_right: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """
    Rectify a raw stereo pair. Returns left_rect, right_rect, K_rect, baseline_m, maps.
    """
    h, w = left.shape[:2]
    if image_size is None:
        image_size = (w, h)
    R = np.asarray(R_left_to_right, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_left_to_right, dtype=np.float64).reshape(3, 1)
    baseline_m = float(np.linalg.norm(t))

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_left, D_left, K_right, D_right, image_size, R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, image_size, cv2.CV_32FC1)
    left_rect = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
    K_rect = P1[:3, :3].astype(np.float64)
    maps = {"map1x": map1x, "map1y": map1y, "map2x": map2x, "map2y": map2y, "Q": Q}
    return left_rect, right_rect, K_rect, baseline_m, maps

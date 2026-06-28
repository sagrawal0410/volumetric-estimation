"""Convert stereo disparity to metric depth."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def disparity_to_depth_m(
    disparity_px: np.ndarray,
    fx_px: float,
    baseline_m: float,
    min_disp: float = 0.1,
    max_depth_m: float = 5.0,
) -> np.ndarray:
    """
    depth_m = fx_px * baseline_m / disparity_px

    Invalid where disparity <= min_disp, non-finite, or depth out of range -> 0.
    """
    disp = disparity_px.astype(np.float32)
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = np.isfinite(disp) & (disp > min_disp)
    depth[valid] = (float(fx_px) * float(baseline_m)) / disp[valid]
    invalid = ~valid | (depth <= 0) | (depth > max_depth_m) | ~np.isfinite(depth)
    depth[invalid] = 0.0
    return depth


def make_depth_valid_mask(depth_m: np.ndarray, object_mask: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0.01)
    if object_mask is not None:
        valid &= object_mask.astype(bool)
    return valid


def save_disparity_debug(disparity: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), disparity.astype(np.float32))
    disp = disparity.copy()
    disp[~np.isfinite(disp)] = 0
    if np.any(disp > 0):
        lo, hi = np.percentile(disp[disp > 0], [5, 95])
        norm = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        norm = disp
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), color)


def save_depth_debug(depth_m: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = depth_m > 0.01
    vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if np.any(valid):
        lo = float(np.percentile(depth_m[valid], 5))
        hi = float(np.percentile(depth_m[valid], 95))
        if hi <= lo:
            hi = lo + 0.01
        norm = np.clip((depth_m - lo) / (hi - lo), 0, 1)
        vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[~valid] = 0
    cv2.imwrite(str(path), vis)

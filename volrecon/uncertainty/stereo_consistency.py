"""Left-right stereo consistency confidence."""

from __future__ import annotations

import numpy as np


def warp_disparity_right_to_left(
    disparity_r2l: np.ndarray,
    disparity_l2r: np.ndarray,
    convention: str = "standard",
) -> np.ndarray:
    """
    Sample right-to-left disparity at left pixel locations.

    convention='standard': u_r = u_l - d_l, compare d_l with d_r(u_r).
    Returns warped d_r at each left pixel (NaN where invalid).
    """
    d_l = np.asarray(disparity_l2r, dtype=np.float64)
    d_r = np.asarray(disparity_r2l, dtype=np.float64)
    h, w = d_l.shape
    warped = np.full((h, w), np.nan, dtype=np.float64)

    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    u_r = u_grid - d_l
    valid = (
        np.isfinite(d_l)
        & (d_l > 0)
        & (u_r >= 0)
        & (u_r <= w - 1)
        & (v_grid >= 0)
        & (v_grid <= h - 1)
    )
    if not np.any(valid):
        return warped

    from scipy.ndimage import map_coordinates

    coords = np.stack([v_grid[valid], u_r[valid]], axis=0)
    sampled = map_coordinates(d_r, coords, order=1, mode="nearest")
    warped[valid] = sampled
    return warped


def lr_consistency_error(disparity_l2r: np.ndarray, disparity_r2l_warped: np.ndarray) -> np.ndarray:
    """Absolute disparity consistency error at left pixels."""
    d_l = np.asarray(disparity_l2r, dtype=np.float64)
    d_r = np.asarray(disparity_r2l_warped, dtype=np.float64)
    err = np.full(d_l.shape, np.inf, dtype=np.float64)
    valid = np.isfinite(d_l) & np.isfinite(d_r) & (d_l > 0) & (d_r > 0)
    err[valid] = np.abs(d_l[valid] - d_r[valid])
    return err


def lr_consistency_confidence(
    disparity_l2r: np.ndarray,
    disparity_r2l: np.ndarray | None,
    tau_lr_px: float = 1.5,
) -> np.ndarray:
    if disparity_r2l is None:
        return np.ones_like(disparity_l2r, dtype=np.float64)
    warped = warp_disparity_right_to_left(disparity_r2l, disparity_l2r)
    err = lr_consistency_error(disparity_l2r, warped)
    c = np.exp(-err / max(tau_lr_px, 1e-6))
    c[~np.isfinite(err)] = 0.0
    return np.clip(c, 0.0, 1.0)


def photometric_warp_right_to_left(right_img: np.ndarray, disparity_l2r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Warp right image to left view; return warped image and valid mask."""
    from scipy.ndimage import map_coordinates

    right = np.asarray(right_img, dtype=np.float64)
    d = np.asarray(disparity_l2r, dtype=np.float64)
    h, w = d.shape
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    u_r = u_grid - d
    valid = np.isfinite(d) & (d > 0) & (u_r >= 0) & (u_r <= w - 1)

    if right.ndim == 3:
        warped = np.zeros((h, w, right.shape[2]), dtype=np.float64)
        for c in range(right.shape[2]):
            coords = np.stack([v_grid, u_r], axis=0)
            warped[..., c] = map_coordinates(right[..., c], coords, order=1, mode="constant", cval=0)
    else:
        coords = np.stack([v_grid, u_r], axis=0)
        warped = map_coordinates(right, coords, order=1, mode="constant", cval=0)

    warped[~valid] = 0.0
    return warped, valid

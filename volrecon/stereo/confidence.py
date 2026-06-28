"""Confidence / validity masks for stereo depth."""

from __future__ import annotations

import numpy as np


def build_valid_depth_mask(
    disparity_px: np.ndarray,
    depth_m: np.ndarray,
    min_disp: float = 0.5,
    min_depth_m: float = 0.1,
    max_depth_m: float = 2.0,
) -> np.ndarray:
    disp = np.asarray(disparity_px, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(disp) & np.isfinite(depth)
    valid &= disp > min_disp
    valid &= depth >= min_depth_m
    valid &= depth <= max_depth_m
    return valid


def apply_depth_mask(depth_m: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = depth_m.astype(np.float64).copy()
    out[~valid_mask] = 0.0
    return out

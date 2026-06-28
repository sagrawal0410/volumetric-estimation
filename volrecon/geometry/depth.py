"""Depth map utilities."""

from __future__ import annotations

import numpy as np

from volrecon.geometry.camera import validate_positive_depth


def depth_uint16_to_meters(depth: np.ndarray, depth_scale: float) -> np.ndarray:
    """Convert raw depth image to meters using BOP-style depth_scale."""
    return depth.astype(np.float64) * float(depth_scale)


def apply_depth_scale(depth: np.ndarray, scale: float, original_units: str = "raw") -> np.ndarray:
    del original_units  # preserved in metadata elsewhere
    return depth.astype(np.float64) * float(scale)


def clip_depth(depth_m: np.ndarray, max_depth_m: float | None = None) -> np.ndarray:
    out = depth_m.astype(np.float64).copy()
    if max_depth_m is not None:
        out[out > max_depth_m] = 0.0
    validate_positive_depth(out, "depth_m")
    return out

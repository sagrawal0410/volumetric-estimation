"""Debug visualization helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import trimesh

from tless_volume_benchmark.geometry import backproject_masked_depth_to_object
from tless_volume_benchmark.io_bop import CandidateFrame


def depth_to_vis(depth_m: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Colorize depth for debug PNG."""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if mask is not None:
        valid &= mask
    vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return vis
    d = depth_m.copy()
    d[~valid] = np.nan
    lo = float(np.nanpercentile(d, 5))
    hi = float(np.nanpercentile(d, 95))
    if hi <= lo:
        hi = lo + 0.01
    norm = np.clip((depth_m - lo) / (hi - lo), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    vis = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    return vis


def save_mask_overlay(path: Path, rgb: np.ndarray, mask: np.ndarray) -> None:
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    color = np.zeros_like(overlay)
    color[mask] = (0, 255, 0)
    blended = cv2.addWeighted(overlay, 0.7, color, 0.3, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), blended)


def save_fused_points_colored(
    path: Path,
    frames: list[CandidateFrame],
) -> np.ndarray:
    """Backproject all frames and save colored PLY."""
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ]
    all_pts = []
    all_cols = []
    for i, frame in enumerate(frames):
        pts = backproject_masked_depth_to_object(
            frame.depth_m, frame.mask, frame.K, frame.T_cam_to_object
        )
        if pts.size:
            all_pts.append(pts)
            col = colors[i % len(colors)]
            all_cols.append(np.tile(col, (pts.shape[0], 1)))
    if not all_pts:
        raise ValueError("No points to fuse for debug visualization")
    points = np.vstack(all_pts)
    point_colors = np.vstack(all_cols)
    cloud = trimesh.PointCloud(vertices=points, colors=point_colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud.export(path)
    return points

"""Scene bounds for TSDF integration."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh

from volrecon.geometry.camera import backproject_depth
from volrecon.geometry.transforms import transform_points
from volrecon.io.json_io import write_json


def compute_bounds_from_camera_frustums(
    Ks: Sequence[np.ndarray],
    T_world_cams: Sequence[np.ndarray],
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Return AABB [min_xyz, max_xyz] from camera frustum corners in world frame."""
    corners_world: list[np.ndarray] = []
    for K, T_wc in zip(Ks, T_world_cams, strict=True):
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        w = int(cx * 2)
        h = int(cy * 2)
        for z in (min_depth_m, max_depth_m):
            us = [0, w - 1]
            vs = [0, h - 1]
            for u in us:
                for v in vs:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    p_cam = np.array([x, y, z, 1.0])
                    p_world = T_wc @ p_cam
                    corners_world.append(p_world[:3])
    pts = np.stack(corners_world, axis=0)
    return np.stack([pts.min(axis=0), pts.max(axis=0)])


def compute_bounds_from_gt_mesh(mesh: trimesh.Trimesh) -> np.ndarray:
    bounds = mesh.bounds
    return np.asarray(bounds, dtype=np.float64)


def compute_bounds_from_depth_points(
    depth_maps: Sequence[np.ndarray],
    Ks: Sequence[np.ndarray],
    T_world_cams: Sequence[np.ndarray],
    sample_stride: int = 8,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> np.ndarray:
    pts_all: list[np.ndarray] = []
    for depth, K, T_wc in zip(depth_maps, Ks, T_world_cams, strict=True):
        d = np.asarray(depth, dtype=np.float64)[::sample_stride, ::sample_stride]
        Ks_sub = np.asarray(K, dtype=np.float64).copy()
        pts_cam = backproject_depth(d, Ks_sub)
        if len(pts_cam) == 0:
            continue
        ones = np.ones((pts_cam.shape[0], 1))
        hom = np.hstack([pts_cam, ones])
        pts_world = (T_wc @ hom.T).T[:, :3]
        pts_all.append(pts_world)
    if not pts_all:
        raise ValueError("No valid depth points for bounds estimation")
    pts = np.vstack(pts_all)
    lo = np.percentile(pts, lower_pct, axis=0)
    hi = np.percentile(pts, upper_pct, axis=0)
    return np.stack([lo, hi])


def robust_expand_bounds(bounds: np.ndarray, margin_m: float) -> np.ndarray:
    b = np.asarray(bounds, dtype=np.float64).copy()
    b[0] -= margin_m
    b[1] += margin_m
    return b


def save_bounds_json(path: Path, bounds: np.ndarray, source: str) -> None:
    write_json(
        path,
        {
            "min_xyz": bounds[0].tolist(),
            "max_xyz": bounds[1].tolist(),
            "source": source,
        },
    )

"""Multi-view depth agreement confidence."""

from __future__ import annotations

import logging

import numpy as np

from volrecon.geometry.camera import backproject_depth, project_points
from volrecon.geometry.transforms import transform_points

logger = logging.getLogger(__name__)


def camera_distance(T_a: np.ndarray, T_b: np.ndarray) -> float:
    ca = T_a[:3, 3]
    cb = T_b[:3, 3]
    return float(np.linalg.norm(ca - cb))


def select_neighbor_views(
    view_index: int,
    T_world_cams: list[np.ndarray],
    k: int,
) -> list[int]:
    if len(T_world_cams) <= 1:
        return []
    dists = [(i, camera_distance(T_world_cams[view_index], T)) for i, T in enumerate(T_world_cams) if i != view_index]
    dists.sort(key=lambda x: x[1])
    return [i for i, _ in dists[:k]]


def multiview_agreement_confidence(
    depth_m: np.ndarray,
    K: np.ndarray,
    T_world_cam: np.ndarray,
    T_cam_world: np.ndarray,
    neighbor_depths: list[np.ndarray],
    neighbor_Ks: list[np.ndarray],
    neighbor_T_cams_world: list[np.ndarray],
    tau_mv_m: float = 0.005,
    sample_stride: int = 4,
) -> np.ndarray:
    """
    For each pixel in reference view, backproject, project to neighbors, compare depths.
    Returns per-pixel agreement confidence (median over neighbors).
    """
    h, w = depth_m.shape
    agreement = np.ones((h, w), dtype=np.float64)

    if not neighbor_depths:
        return agreement

    d = depth_m[::sample_stride, ::sample_stride]
    K = np.asarray(K, dtype=np.float64)
    errors_all: list[np.ndarray] = []

    pts_cam = backproject_depth(d, K)
    if len(pts_cam) == 0:
        return agreement

    ones = np.ones((pts_cam.shape[0], 1))
    pts_world = transform_points(T_world_cam, pts_cam)

    for nd, nK, nT_cw in zip(neighbor_depths, neighbor_Ks, neighbor_T_cams_world, strict=True):
        nK = np.asarray(nK, dtype=np.float64)
        nT_cw = np.asarray(nT_cw, dtype=np.float64)
        pts_n = transform_points(nT_cw, pts_world)
        uv, z_proj = project_points(pts_n, nK)
        nh, nw = nd.shape
        errs = np.full(len(pts_cam), np.nan, dtype=np.float64)
        for i, (u, v, z) in enumerate(zip(uv[:, 0], uv[:, 1], z_proj, strict=True)):
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < nw and 0 <= vi < nh and z > 0:
                z_n = nd[vi, ui]
                if z_n > 0:
                    errs[i] = abs(z_n - z)
        errors_all.append(errs)

    if not errors_all:
        return agreement

    err_stack = np.stack(errors_all, axis=0)
    median_err = np.nanmedian(err_stack, axis=0)
    c_sub = np.exp(-median_err / max(tau_mv_m, 1e-6))
    c_sub[~np.isfinite(median_err)] = 1.0

    agreement_sub = np.ones((h // sample_stride + (1 if h % sample_stride else 0),
                             w // sample_stride + (1 if w % sample_stride else 0)), dtype=np.float64)
    sh, sw = c_sub.shape
    agreement_sub[:sh, :sw] = c_sub

    # Upsample back
    import cv2

    agreement = cv2.resize(agreement_sub[: h // sample_stride, : w // sample_stride], (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(agreement, 0.0, 1.0)

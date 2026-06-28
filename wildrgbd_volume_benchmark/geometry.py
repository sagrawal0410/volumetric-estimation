"""Geometry helpers for WildRGB-D volume estimation."""

from __future__ import annotations

import numpy as np


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    hom = np.hstack([points, ones])
    return (T @ hom.T).T[:, :3]


def backproject_depth(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T_cam_to_world: np.ndarray,
    depth_min: float = 0.05,
    depth_max: float = 5.0,
) -> np.ndarray:
    """Backproject masked depth to world coordinates (OpenCV convention)."""
    valid = mask & np.isfinite(depth_m) & (depth_m >= depth_min) & (depth_m <= depth_max)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    v_idx, u_idx = np.where(valid)
    z = depth_m[valid].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = u_idx.astype(np.float64)
    v = v_idx.astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)
    R = T_cam_to_world[:3, :3]
    t = T_cam_to_world[:3, 3]
    return (R @ pts_cam.T).T + t


def backproject_depth_to_object(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T_cam_to_object: np.ndarray,
    depth_min: float = 0.05,
    depth_max: float = 5.0,
) -> np.ndarray:
    valid = mask & np.isfinite(depth_m) & (depth_m >= depth_min) & (depth_m <= depth_max)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    v_idx, u_idx = np.where(valid)
    z = depth_m[valid].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = u_idx.astype(np.float64)
    v = v_idx.astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)
    return transform_points(pts_cam, T_cam_to_object)


def project_object_points_to_camera(
    points_obj: np.ndarray,
    K: np.ndarray,
    T_object_to_cam: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_shape
    pts_cam = transform_points(points_obj, T_object_to_cam)
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    in_front = z > 1e-6
    u = K[0, 0] * x / np.maximum(z, 1e-9) + K[0, 2]
    v = K[1, 1] * y / np.maximum(z, 1e-9) + K[1, 2]
    in_image = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, z, in_image


def make_o3d_intrinsic(K: np.ndarray, width: int, height: int):
    import open3d as o3d

    return o3d.camera.PinholeCameraIntrinsic(
        width=int(width),
        height=int(height),
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )


def estimate_object_frame_from_full_points(
    points_world: np.ndarray,
    use_median: bool = True,
) -> np.ndarray:
    """T_world_to_object: translate so object centroid is at origin (identity rotation)."""
    if points_world.shape[0] == 0:
        return np.eye(4, dtype=np.float64)
    pts = points_world.copy()
    if pts.shape[0] > 100:
        lo = np.percentile(pts, 2, axis=0)
        hi = np.percentile(pts, 98, axis=0)
        keep = np.all((pts >= lo) & (pts <= hi), axis=1)
        if np.any(keep):
            pts = pts[keep]
    centroid = np.median(pts, axis=0) if use_median else pts.mean(axis=0)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = -centroid
    return T


def estimate_bounds(points_obj: np.ndarray, padding: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    if points_obj.shape[0] == 0:
        raise ValueError("Cannot estimate bounds from empty point cloud")
    lo = points_obj.min(axis=0) - padding
    hi = points_obj.max(axis=0) + padding
    return lo, hi


def camera_center_from_T(T_cam_to_ref: np.ndarray) -> np.ndarray:
    """Camera center in reference frame (world or object)."""
    return T_cam_to_ref[:3, 3].copy()

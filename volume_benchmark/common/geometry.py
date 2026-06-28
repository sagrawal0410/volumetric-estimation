"""Camera geometry helpers (OpenCV convention: x right, y down, z forward)."""

from __future__ import annotations

import numpy as np


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 4x4 rigid transform from rotation (3,3) and translation (3,) or (3,1)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3), got {R.shape}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must have shape (4, 4), got {T.shape}")
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = -R_inv @ t
    return T_inv


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to Nx3 points."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    T = np.asarray(T, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homo = np.hstack([points, ones])
    transformed = (T @ homo.T).T
    return transformed[:, :3]


def convert_mm_pose_to_meters(T_mm: np.ndarray) -> np.ndarray:
    """Convert a pose with millimeter translation to meters."""
    T = np.asarray(T_mm, dtype=np.float64).copy()
    if T.shape != (4, 4):
        raise ValueError(f"T_mm must have shape (4, 4), got {T.shape}")
    T[:3, 3] /= 1000.0
    return T


def backproject_depth_to_object(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T_cam_to_object: np.ndarray,
    depth_min: float = 0.05,
    depth_max: float = 5.0,
) -> np.ndarray:
    """
    Back-project masked depth pixels into the object coordinate frame.

    Returns an (N, 3) array of 3-D points in object coordinates (meters).
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    K = np.asarray(K, dtype=np.float64)
    T_cam_to_object = np.asarray(T_cam_to_object, dtype=np.float64)

    if depth_m.shape != mask.shape:
        raise ValueError(
            f"depth and mask shape mismatch: {depth_m.shape} vs {mask.shape}"
        )

    valid = mask & (depth_m > depth_min) & (depth_m < depth_max) & np.isfinite(depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    v_idx, u_idx = np.nonzero(valid)
    z = depth_m[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (u_idx.astype(np.float64) - cx) * z / fx
    y_cam = (v_idx.astype(np.float64) - cy) * z / fy
    points_cam = np.stack([x_cam, y_cam, z], axis=1)

    return transform_points(points_cam, T_cam_to_object)


def project_object_points_to_camera(
    points_object: np.ndarray,
    K: np.ndarray,
    T_object_to_cam: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project object-frame points into the camera image.

    Parameters
    ----------
    points_object : (N, 3) points in object coordinates (meters).
    K : (3, 3) intrinsic matrix.
    T_object_to_cam : (4, 4) transform from object to camera frame.
    image_shape : (height, width).

    Returns
    -------
    u, v, depth_m : pixel coordinates and camera-frame depth for valid projections.
    """
    points_object = np.asarray(points_object, dtype=np.float64)
    if points_object.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty

    K = np.asarray(K, dtype=np.float64)
    T_object_to_cam = np.asarray(T_object_to_cam, dtype=np.float64)
    height, width = image_shape

    points_cam = transform_points(points_object, T_object_to_cam)
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    in_front = z > 1e-6
    if not np.any(in_front):
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty

    x, y, z = x[in_front], y[in_front], z[in_front]

    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]

    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[in_image], v[in_image], z[in_image]

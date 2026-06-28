"""Geometry helpers for T-LESS / BOP object-centric volume estimation."""

from __future__ import annotations

import numpy as np


def make_T(R: np.ndarray, t_m: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from 3x3 rotation and 3-vector translation (meters)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t_m, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def bop_pose_m2c_to_T_cam_to_object(cam_R_m2c, cam_t_m2c_mm) -> np.ndarray:
    """
    Convert BOP model-to-camera pose to T_cam_to_object.

    BOP stores cam_t_m2c in millimeters; model vertices are also in mm but
    are converted to meters elsewhere. Translation here is converted to meters.
    """
    R = np.asarray(cam_R_m2c, dtype=np.float64).reshape(3, 3)
    t_m = np.asarray(cam_t_m2c_mm, dtype=np.float64).reshape(3) / 1000.0
    T_m2c = make_T(R, t_m)
    return invert_T(T_m2c)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    hom = np.hstack([points, ones])
    out = (T @ hom.T).T
    return out[:, :3]


def backproject_masked_depth_to_object(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T_cam_to_object: np.ndarray,
    depth_min: float = 0.05,
    depth_max: float = 5.0,
) -> np.ndarray:
    """
    Backproject masked depth pixels to object coordinates.

    Uses OpenCV camera convention: x right, y down, z forward.
    """
    valid = mask & np.isfinite(depth_m) & (depth_m >= depth_min) & (depth_m <= depth_max)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)

    v_idx, u_idx = np.where(valid)
    z = depth_m[valid].astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
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
    """
    Project object-frame points to pixel coordinates.

    Returns (u, v, z_cam, in_image) where in_image is bool mask.
    image_shape is (height, width).
    """
    h, w = image_shape
    pts_cam = transform_points(points_obj, T_object_to_cam)
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    in_front = z > 1e-6
    u = K[0, 0] * x / np.maximum(z, 1e-9) + K[0, 2]
    v = K[1, 1] * y / np.maximum(z, 1e-9) + K[1, 2]
    in_image = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, z, in_image


def create_o3d_intrinsic_from_K(K: np.ndarray, width: int, height: int):
    """Create Open3D PinholeCameraIntrinsic from 3x3 K."""
    import open3d as o3d

    return o3d.camera.PinholeCameraIntrinsic(
        width=int(width),
        height=int(height),
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )


def camera_center_object(T_cam_to_object: np.ndarray) -> np.ndarray:
    """Camera center expressed in object coordinates."""
    return T_cam_to_object[:3, 3].copy()

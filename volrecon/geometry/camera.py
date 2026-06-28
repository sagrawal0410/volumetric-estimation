"""Camera geometry utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volrecon.geometry.transforms import transform_points


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    K: np.ndarray
    distortion: list[float] | None = None

    @classmethod
    def from_K(cls, K: np.ndarray, width: int, height: int, distortion: list[float] | None = None) -> "CameraIntrinsics":
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        return cls(
            width=width,
            height=height,
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            K=K.copy(),
            distortion=distortion,
        )


def K_to_intrinsics(K: np.ndarray, width: int, height: int, distortion: list[float] | None = None) -> CameraIntrinsics:
    return CameraIntrinsics.from_K(K, width, height, distortion)


def resize_intrinsics(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64).copy()
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y
    return K


def backproject_depth(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Backproject a depth map (meters) to Nx3 camera-frame points."""
    depth_m = np.asarray(depth_m, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_m.reshape(-1)
    valid = z > 0
    u = u.reshape(-1)[valid].astype(np.float64)
    v = v.reshape(-1)[valid].astype(np.float64)
    z = z[valid]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 camera-frame points to pixel coords and depths."""
    pts = np.asarray(points_cam, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    z = pts[:, 2]
    uv_h = (K @ pts.T).T
    uv = uv_h[:, :2] / np.maximum(uv_h[:, 2:3], 1e-12)
    return uv, z


def depth_to_pointcloud(
    depth_m: np.ndarray,
    K: np.ndarray,
    rgb: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    depth_m = np.asarray(depth_m, dtype=np.float64)
    if mask is not None:
        depth_m = depth_m.copy()
        depth_m[~mask] = 0.0
    points = backproject_depth(depth_m, K)
    colors = None
    if rgb is not None:
        rgb = np.asarray(rgb)
        h, w = depth_m.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_m.reshape(-1)
        valid = z > 0
        if mask is not None:
            valid &= mask.reshape(-1)
        colors = rgb.reshape(-1, rgb.shape[-1])[valid].astype(np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
    return points, colors


def disparity_to_depth(disparity_px: np.ndarray | float, fx_px: float, baseline_m: float) -> np.ndarray | float:
    """depth = fx * baseline / disparity (meters)."""
    disp = np.asarray(disparity_px, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = fx_px * baseline_m / disp
    if np.isscalar(disparity_px):
        return float(depth)
    depth[disp <= 0] = 0.0
    return depth


def validate_positive_depth(depth_m: np.ndarray, name: str = "depth") -> None:
    depth_m = np.asarray(depth_m)
    if np.any(depth_m[depth_m > 0] <= 0):
        raise ValueError(f"{name} contains non-positive values among valid pixels")

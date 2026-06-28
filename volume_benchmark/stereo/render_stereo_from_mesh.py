"""Render rectified synthetic stereo pairs from a GT mesh."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import trimesh

from volume_benchmark.common.geometry import invert_T, transform_points

TextureMode = Literal["flat", "mesh_texture"]


def _rasterize_mesh(
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    T_cam_to_object: np.ndarray,
    image_size: tuple[int, int],
    flat_color: tuple[int, int, int] = (180, 180, 180),
) -> tuple[np.ndarray, np.ndarray]:
    """Simple z-buffer rasterization in camera frame."""
    height, width = image_size
    T_object_to_cam = invert_T(T_cam_to_object)
    mesh_cam = mesh.copy()
    mesh_cam.apply_transform(T_object_to_cam)

    try:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh_cam)
    except Exception:
        surface, _ = trimesh.sample.sample_surface(mesh, 5000)
        pts_cam = transform_points(surface, T_object_to_cam)
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=bool)
        depth = np.zeros((height, width), dtype=np.float32)
        z = pts_cam[:, 2]
        valid = z > 0.01
        u = (K[0, 0] * pts_cam[valid, 0] / z[valid] + K[0, 2]).astype(int)
        v = (K[1, 1] * pts_cam[valid, 1] / z[valid] + K[1, 2]).astype(int)
        for ui, vi, zi in zip(u, v, z[valid]):
            if 0 <= ui < width and 0 <= vi < height:
                if depth[vi, ui] == 0 or zi < depth[vi, ui]:
                    depth[vi, ui] = zi
                    mask[vi, ui] = True
                    rgb[vi, ui] = flat_color
        return rgb, mask

    v_coords, u_coords = np.mgrid[0:height, 0:width]
    u = u_coords.astype(np.float64).ravel()
    v = v_coords.astype(np.float64).ravel()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.zeros_like(dirs)
    locations, index_ray, _ = intersector.intersects_location(origins, dirs, multiple_hits=False)

    rgb = np.zeros((height * width, 3), dtype=np.uint8)
    mask = np.zeros(height * width, dtype=bool)
    depth = np.zeros(height * width, dtype=np.float32)
    if len(index_ray):
        depth[index_ray] = locations[:, 2].astype(np.float32)
        mask[index_ray] = True
        rgb[index_ray] = flat_color
    return rgb.reshape(height, width, 3), mask.reshape(height, width)


def _right_pose_from_left(T_left_cam_to_object: np.ndarray, baseline_m: float) -> np.ndarray:
    """Rectified stereo: same rotation, right camera shifted +baseline along left +x."""
    T = np.asarray(T_left_cam_to_object, dtype=np.float64).copy()
    R = T[:3, :3]
    t = T[:3, 3]
    t_right = t + R @ np.array([baseline_m, 0.0, 0.0], dtype=np.float64)
    T_right = np.eye(4, dtype=np.float64)
    T_right[:3, :3] = R
    T_right[:3, 3] = t_right
    return T_right


def render_rectified_stereo_from_mesh(
    mesh_path: str | Path,
    K: np.ndarray,
    image_size: tuple[int, int],
    T_left_cam_to_object: np.ndarray,
    baseline_m: float,
    texture_mode: TextureMode = "flat",
    background: str = "plain",
    object_mask: bool = True,
    mesh_units: str = "m",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    """
    Render rectified left/right RGB and optional left mask from a mesh.

    Returns left_rgb, right_rgb, left_mask (or None), metadata.
    """
    mesh_path = Path(mesh_path)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected mesh at {mesh_path}")
    if mesh_units == "mm":
        mesh = mesh.copy()
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) / 1000.0

    width, height = image_size
    color = (200, 160, 120) if texture_mode == "flat" else (180, 180, 180)
    if background != "plain":
        pass  # keep black background

    T_right = _right_pose_from_left(T_left_cam_to_object, baseline_m)
    left_rgb, left_mask = _rasterize_mesh(mesh, K, T_left_cam_to_object, (height, width), color)
    right_rgb, _ = _rasterize_mesh(mesh, K, T_right, (height, width), color)

    meta: dict[str, Any] = {
        "source_mode": "rendered_stereo_from_gt_mesh",
        "baseline_m": float(baseline_m),
        "texture_mode": texture_mode,
        "mesh_path": str(mesh_path.resolve()),
        "T_right_cam_to_object": T_right.tolist(),
    }
    mask_out = left_mask if object_mask else None
    return left_rgb, right_rgb, mask_out, meta

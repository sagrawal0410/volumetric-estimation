"""Synthetic scan fixtures for tests."""

from __future__ import annotations

import os

os.environ.setdefault("VOLUME_BENCHMARK_SKIP_OPEN3D", "1")

from pathlib import Path

import cv2
import numpy as np
import trimesh

from volume_benchmark.common.geometry import invert_T, make_T
from volume_benchmark.common.io import Frame, save_prepared_scan
from volume_benchmark.common.mesh_volume import compute_mesh_volume_m3, write_gt_volume_json


def make_box_mesh(size: float = 0.2) -> trimesh.Trimesh:
    """Axis-aligned box centered at origin with edge length `size` (meters)."""
    return trimesh.creation.box(extents=(size, size, size))


def _look_at_pose(eye: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Return T_cam_to_object for a camera at `eye` looking at `target` (OpenCV convention)."""
    target = np.zeros(3) if target is None else np.asarray(target, dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64)

    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-12
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, world_up)
    if np.linalg.norm(right) < 1e-9:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], axis=0)
    t = -R @ eye
    return make_T(R, t)


def render_depth_mask(
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    T_cam_to_object: np.ndarray,
    image_size: tuple[int, int] = (128, 128),
) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast render depth (m) and object mask for a watertight mesh."""
    height, width = image_size
    T_object_to_cam = invert_T(T_cam_to_object)
    mesh_cam = mesh.copy()
    mesh_cam.apply_transform(T_object_to_cam)

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh_cam)
    v_coords, u_coords = np.mgrid[0:height, 0:width]
    u = u_coords.astype(np.float64).ravel()
    v = v_coords.astype(np.float64).ravel()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.zeros_like(dirs)

    locations, index_ray, _ = intersector.intersects_location(
        origins, dirs, multiple_hits=False
    )

    depth = np.zeros(height * width, dtype=np.float32)
    mask = np.zeros(height * width, dtype=bool)
    if len(index_ray):
        depth[index_ray] = locations[:, 2].astype(np.float32)
        mask[index_ray] = True

    return depth.reshape(height, width), mask.reshape(height, width)


def create_synthetic_scan(
    output_dir: Path,
    box_size: float = 0.2,
    num_views: int = 5,
    image_size: tuple[int, int] = (128, 128),
) -> Path:
    """Write a prepared scan of a rendered box to disk."""
    return create_shape_scan(
        output_dir,
        mesh=make_box_mesh(box_size),
        num_views=num_views,
        image_size=image_size,
        label="box",
        box_size_m=box_size,
    )


def _fill_silhouette(depth_m: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Close/fill rasterized masks and inpaint missing depth inside silhouette."""
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    try:
        from scipy import ndimage

        mask_bool = ndimage.binary_fill_holes(mask_u8 > 0)
    except Exception:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask_u8)
        if contours:
            cv2.fillPoly(filled, contours, 255)
        mask_bool = filled > 0

    depth = depth_m.copy()
    valid = mask_bool & (depth > 0.01)
    if np.any(valid):
        fill_val = float(np.min(depth[valid]))
        depth[mask_bool & ~valid] = fill_val
    return depth.astype(np.float32), mask_bool


def _rasterize_points_to_depth(
    points_object: np.ndarray,
    K: np.ndarray,
    T_cam_to_object: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer rasterize object-frame points into depth (m) and bool mask."""
    from volume_benchmark.common.geometry import transform_points

    height, width = image_shape
    T_object_to_cam = invert_T(T_cam_to_object)
    points_cam = transform_points(points_object, T_object_to_cam)
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    in_front = z > 1e-4
    depth = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    if not np.any(in_front):
        return depth, mask

    x, y, z = x[in_front], y[in_front], z[in_front]
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    in_image = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui, vi, z = ui[in_image], vi[in_image], z[in_image]
    for uu, vv, zz in zip(ui, vi, z):
        if not mask[vv, uu] or zz < depth[vv, uu]:
            depth[vv, uu] = zz
            mask[vv, uu] = True
    return depth, mask


def create_shape_scan(
    output_dir: Path,
    mesh: trimesh.Trimesh,
    num_views: int = 5,
    image_size: tuple[int, int] = (128, 128),
    label: str = "shape",
    gt_volume_m3: float | None = None,
    **metadata_extra,
) -> Path:
    """Write a prepared scan by rasterizing surface samples (no rtree needed)."""
    if gt_volume_m3 is None:
        volume_m3, watertight, gt_type = compute_mesh_volume_m3(mesh)
    else:
        volume_m3 = float(gt_volume_m3)
        watertight = False
        gt_type = "full_reconstruction_pseudo_gt"
    mesh_path = output_dir / f"_source_{label}.ply"
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(mesh_path)

    width, height = image_size
    K = np.array(
        [[200.0, 0.0, width / 2], [0.0, 200.0, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    surface_points, _ = trimesh.sample.sample_surface(mesh, 4000)
    radius = 0.55
    frames: list[Frame] = []
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        eye = np.array([radius * np.cos(angle), 0.05 * (i % 2), radius * np.sin(angle)])
        T = _look_at_pose(eye)
        depth_m, mask = _rasterize_points_to_depth(
            surface_points, K, T, image_shape=(height, width)
        )
        depth_m, mask = _fill_silhouette(depth_m, mask)
        frames.append(
            Frame(
                depth_m=depth_m,
                mask=mask,
                T_cam_to_object=T,
                source_info={"synthetic": True, "view": i, "shape": label},
            )
        )

    meta = {"synthetic": True, "shape": label, **metadata_extra}
    save_prepared_scan(output_dir, K, frames, mesh_path, metadata=meta)
    write_gt_volume_json(
        output_dir / "gt_volume.json",
        volume_m3=volume_m3,
        method=gt_type,
        watertight=watertight,
        source_mesh=mesh_path,
    )
    return output_dir


def create_bop_like_scan(output_dir: Path, radius: float = 0.05) -> Path:
    """Minimal BOP-style prepared scan (icosphere)."""
    mesh = trimesh.creation.icosphere(radius=radius)
    return create_shape_scan(output_dir, mesh=mesh, num_views=5, label="bop_sphere")


def create_concave_scan(output_dir: Path) -> Path:
    """Two-offset boxes fused — concave L-ish shape, visual hull overestimates."""
    a = trimesh.creation.box(extents=[0.12, 0.12, 0.06])
    b = trimesh.creation.box(extents=[0.06, 0.12, 0.12])
    b.apply_translation([0.03, 0.0, 0.03])
    mesh = trimesh.util.concatenate([a, b])
    gt_m3 = (0.12 * 0.12 * 0.06) + (0.06 * 0.12 * 0.12) - (0.06 * 0.12 * 0.06)
    return create_shape_scan(
        output_dir, mesh=mesh, num_views=6, label="concave_l", gt_volume_m3=gt_m3
    )

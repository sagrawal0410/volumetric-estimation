"""Voxel carving / visual hull on sparse WildRGB-D views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from wildrgbd_volume_benchmark.geometry import backproject_depth_to_object, invert_T, transform_points
from wildrgbd_volume_benchmark.scan_io import (
    load_prepared_scene,
    pseudo_gt_comparison_fields,
    resolve_output_dir,
    write_report,
)


def estimate_voxel_carving_volume(
    prepared_scene_dir: str | Path,
    voxel_size: float = 0.004,
    depth_tolerance: float = 0.010,
    padding: float = 0.02,
    min_views_checked: int = 2,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    scene = load_prepared_scene(prepared_scene_dir)
    out = resolve_output_dir(scene.scene_dir, "voxel_carving", Path(output_dir) if output_dir else None)
    out.mkdir(parents=True, exist_ok=True)

    chunks = []
    for frame in scene.frames:
        pts = backproject_depth_to_object(frame.depth_m, frame.mask, frame.K, frame.T_cam_to_object)
        if pts.size:
            chunks.append(pts)
    points = np.vstack(chunks)
    lo, hi = points.min(axis=0) - padding, points.max(axis=0) + padding

    xs = np.arange(lo[0], hi[0], voxel_size)
    ys = np.arange(lo[1], hi[1], voxel_size)
    zs = np.arange(lo[2], hi[2], voxel_size)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    kept = np.ones(centers.shape[0], dtype=bool)
    views_checked = np.zeros(centers.shape[0], dtype=np.int32)

    for frame in scene.frames:
        T_o2c = invert_T(frame.T_cam_to_object)
        h, w = frame.depth_m.shape
        pts_cam = transform_points(centers, T_o2c)
        x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        in_front = z > 1e-6
        u = frame.K[0, 0] * x / np.maximum(z, 1e-9) + frame.K[0, 2]
        v = frame.K[1, 1] * y / np.maximum(z, 1e-9) + frame.K[1, 2]
        in_image = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ok_idx = np.where(in_image)[0]
        if ok_idx.size == 0:
            continue
        ui = np.clip(np.round(u[in_image]).astype(int), 0, w - 1)
        vi = np.clip(np.round(v[in_image]).astype(int), 0, h - 1)
        z_ok = z[in_image]
        views_checked[ok_idx] += 1
        in_mask = frame.mask[vi, ui]
        depth_px = frame.depth_m[vi, ui]
        valid_depth = np.isfinite(depth_px) & (depth_px > 0.01)
        carve = (~in_mask) | (in_mask & valid_depth & (z_ok < (depth_px - depth_tolerance)))
        kept[ok_idx[carve]] = False

    kept &= views_checked >= min_views_checked
    kept_centers = centers[kept]
    if kept_centers.shape[0] == 0:
        raise ValueError("Voxel carving removed all voxels")

    volume_m3 = float(kept_centers.shape[0]) * (voxel_size ** 3)
    trimesh.PointCloud(kept_centers).export(out / "carved_voxels.ply")
    np.savez_compressed(out / "carved_voxels.npz", centers_m=kept_centers.astype(np.float32), voxel_size=voxel_size)

    report: dict[str, Any] = {
        "method": "voxel_carving",
        "prepared_scene_dir": str(scene.scene_dir),
        "volume_m3": volume_m3,
        "volume_cm3": volume_m3 * 1e6,
        "voxel_size": voxel_size,
        "expected_bias": "visual hull usually overestimates concave objects",
        "status": "ok",
    }
    report.update(pseudo_gt_comparison_fields(scene, volume_m3))
    write_report(out / "report.json", report)
    return report

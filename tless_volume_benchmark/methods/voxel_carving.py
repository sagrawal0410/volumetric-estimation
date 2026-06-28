"""Depth-aware voxel carving volume estimation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from tless_volume_benchmark.geometry import backproject_masked_depth_to_object, invert_T, transform_points
from tless_volume_benchmark.scan_io import (
    gt_comparison_fields,
    load_prepared_scan,
    resolve_output_dir,
    write_report,
)


def _object_bounds(scan, padding: float) -> tuple[np.ndarray, np.ndarray]:
    chunks = []
    for frame in scan.frames:
        pts = backproject_masked_depth_to_object(
            frame.depth_m, frame.mask, frame.K, frame.T_cam_to_object
        )
        if pts.size:
            chunks.append(pts)
    if not chunks:
        raise ValueError("Could not estimate object bounds")
    points = np.vstack(chunks)
    return points.min(axis=0) - padding, points.max(axis=0) + padding


def _carve_frame(centers, kept, views_checked, frame, depth_tolerance):
    T_object_to_cam = invert_T(frame.T_cam_to_object)
    h, w = frame.depth_m.shape
    depth = frame.depth_m
    mask = frame.mask

    pts_cam = transform_points(centers, T_object_to_cam)
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    in_front = z > 1e-6
    u = frame.K[0, 0] * x / np.maximum(z, 1e-9) + frame.K[0, 2]
    v = frame.K[1, 1] * y / np.maximum(z, 1e-9) + frame.K[1, 2]
    in_image = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ok_idx = np.where(in_image)[0]
    if ok_idx.size == 0:
        return

    ui = np.clip(np.round(u[in_image]).astype(int), 0, w - 1)
    vi = np.clip(np.round(v[in_image]).astype(int), 0, h - 1)
    z_ok = z[in_image]
    views_checked[ok_idx] += 1

    in_mask = mask[vi, ui]
    depth_px = depth[vi, ui]
    valid_depth = np.isfinite(depth_px) & (depth_px > 0.01)

    carve = np.zeros(ok_idx.size, dtype=bool)
    carve |= ~in_mask
    carve |= in_mask & valid_depth & (z_ok < (depth_px - depth_tolerance))
    kept[ok_idx[carve]] = False


def estimate_voxel_carving(
    scan_dir: str | Path,
    voxel_size: float = 0.0025,
    depth_tolerance: float = 0.0075,
    padding: float = 0.015,
    min_views_checked: int = 2,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    scan = load_prepared_scan(scan_dir)
    out = resolve_output_dir(scan.scan_dir, "voxel_carving", Path(output_dir) if output_dir else None)
    out.mkdir(parents=True, exist_ok=True)

    lo, hi = _object_bounds(scan, padding)
    xs = np.arange(lo[0], hi[0], voxel_size)
    ys = np.arange(lo[1], hi[1], voxel_size)
    zs = np.arange(lo[2], hi[2], voxel_size)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    num_total = centers.shape[0]
    max_voxels = int(os.environ.get("TLESS_MAX_VOXELS", "50000000"))
    if num_total > max_voxels:
        extent = hi - lo
        raise ValueError(
            f"Voxel grid too large ({num_total:,} voxels, limit {max_voxels:,}). "
            f"Object bounds extent (m): {extent.tolist()}, voxel_size={voxel_size}. "
            "This often means depth is in mm instead of meters, or voxel_size is too small. "
            "Try larger --voxel_size (e.g. 0.004) or re-run tless_prepare."
        )

    kept = np.ones(num_total, dtype=bool)
    views_checked = np.zeros(num_total, dtype=np.int32)
    for frame in scan.frames:
        _carve_frame(centers, kept, views_checked, frame, depth_tolerance)
        if not kept.any():
            break

    kept &= views_checked >= min_views_checked
    kept_centers = centers[kept]
    if kept_centers.shape[0] == 0:
        raise ValueError("Voxel carving removed all voxels")

    volume_m3 = float(kept_centers.shape[0]) * (voxel_size ** 3)
    trimesh.PointCloud(kept_centers).export(out / "carved_voxels.ply")
    np.savez_compressed(out / "carved_voxels.npz", centers_m=kept_centers.astype(np.float32), voxel_size=voxel_size)

    report: dict[str, Any] = {
        "method": "voxel_carving",
        "scan_dir": str(scan.scan_dir),
        "volume_m3": volume_m3,
        "volume_cm3": volume_m3 * 1e6,
        "voxel_size": voxel_size,
        "depth_tolerance": depth_tolerance,
        "min_views_checked": min_views_checked,
        "num_voxels_total": num_total,
        "num_voxels_kept": int(kept_centers.shape[0]),
        "expected_bias": "visual hull usually overestimates concave objects",
        "outputs": {
            "carved_voxels_ply": str(out / "carved_voxels.ply"),
            "carved_voxels_npz": str(out / "carved_voxels.npz"),
        },
    }
    report.update(gt_comparison_fields(scan, volume_m3))
    write_report(out / "report.json", report)
    return report

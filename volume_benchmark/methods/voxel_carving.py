"""Visual hull / depth-aware voxel carving volume estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import trimesh

from volume_benchmark.common.geometry import backproject_depth_to_object, invert_T, transform_points
from volume_benchmark.common.io import Frame
from volume_benchmark.methods._io import (
    gt_comparison_fields,
    load_scan_or_raise,
    resolve_output_dir,
    write_report,
)


def _object_bounds_from_frames(
    frames: Sequence[Frame],
    K: np.ndarray,
    padding: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    points_list = []
    for frame in frames:
        pts = backproject_depth_to_object(
            frame.depth_m, frame.mask, K, frame.T_cam_to_object
        )
        if pts.size:
            points_list.append(pts)
    if not points_list:
        raise ValueError("Could not estimate object bounds: no valid depth points")
    points = np.vstack(points_list)
    lo = points.min(axis=0) - padding
    hi = points.max(axis=0) + padding
    return lo, hi


def _carve_with_frame(
    centers: np.ndarray,
    kept: np.ndarray,
    views_checked: np.ndarray,
    frame: Frame,
    K: np.ndarray,
    depth_tolerance: float,
) -> None:
    """Apply one-view carving rules in place."""
    T_object_to_cam = invert_T(frame.T_cam_to_object)
    h, w = frame.depth_m.shape
    depth = frame.depth_m
    mask = frame.mask

    points_cam = transform_points(centers, T_object_to_cam)
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    in_front = z > 1e-6
    u = K[0, 0] * x / np.maximum(z, 1e-9) + K[0, 2]
    v = K[1, 1] * y / np.maximum(z, 1e-9) + K[1, 2]
    in_image = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ok = in_front & in_image

    ok_idx = np.where(ok)[0]
    if ok_idx.size == 0:
        return

    ui = np.clip(np.round(u[ok]).astype(int), 0, w - 1)
    vi = np.clip(np.round(v[ok]).astype(int), 0, h - 1)
    z_ok = z[ok]
    views_checked[ok_idx] += 1

    in_mask = mask[vi, ui]
    depth_px = depth[vi, ui]
    valid_depth = np.isfinite(depth_px) & (depth_px > 0.01)

    carve = np.zeros(ok_idx.size, dtype=bool)
    # Outside mask -> carve
    carve |= ~in_mask
    # Inside mask with valid depth: carve if voxel is in front of surface
    front = in_mask & valid_depth & (z_ok < (depth_px - depth_tolerance))
    carve |= front

    kept[ok_idx[carve]] = False


def estimate_volume_voxel_carving(
    frames: Sequence[Frame],
    K: np.ndarray,
    voxel_size: float = 0.004,
    depth_tolerance: float = 0.010,
    padding: float = 0.02,
    min_views_checked: int = 2,
) -> tuple[float, np.ndarray, int, int]:
    """
    Depth-aware voxel carving in object coordinates.

    Returns (volume_m3, kept_centers, num_total, num_kept).
    """
    if not frames:
        raise ValueError("At least one frame is required")
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")

    lo, hi = _object_bounds_from_frames(frames, K, padding=padding)
    xs = np.arange(lo[0], hi[0], voxel_size)
    ys = np.arange(lo[1], hi[1], voxel_size)
    zs = np.arange(lo[2], hi[2], voxel_size)
    if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
        raise ValueError("Voxel grid is empty; check bounds and voxel_size")

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    num_total = centers.shape[0]

    kept = np.ones(num_total, dtype=bool)
    views_checked = np.zeros(num_total, dtype=np.int32)

    for frame in frames:
        _carve_with_frame(centers, kept, views_checked, frame, K, depth_tolerance)
        if not kept.any():
            break

    kept &= views_checked >= min_views_checked
    kept_centers = centers[kept]
    num_kept = int(kept_centers.shape[0])
    if num_kept == 0:
        raise ValueError("Voxel carving removed all voxels; check masks, poses, and thresholds")

    volume_m3 = num_kept * (voxel_size ** 3)
    return volume_m3, kept_centers, num_total, num_kept


def estimate_voxel_carving_volume(
    scan_dir: str | Path,
    output_dir: str | Path | None = None,
    voxel_size: float = 0.004,
    depth_tolerance: float = 0.010,
    padding: float = 0.02,
    min_views_checked: int = 2,
) -> dict[str, Any]:
    """Run voxel carving on a prepared scan and write voxel outputs + report."""
    scan = load_scan_or_raise(scan_dir)
    out = resolve_output_dir(scan.scan_dir, "voxel_carving", output_dir)
    out.mkdir(parents=True, exist_ok=True)

    volume_m3, kept_centers, num_total, num_kept = estimate_volume_voxel_carving(
        scan.frames,
        scan.K,
        voxel_size=voxel_size,
        depth_tolerance=depth_tolerance,
        padding=padding,
        min_views_checked=min_views_checked,
    )

    trimesh.PointCloud(kept_centers).export(out / "carved_voxels.ply")
    np.savez_compressed(
        out / "carved_voxels.npz",
        centers_m=kept_centers.astype(np.float32),
        voxel_size=voxel_size,
    )

    report: dict[str, Any] = {
        "method": "voxel_carving",
        "scan_dir": str(scan.scan_dir),
        "volume_m3": volume_m3,
        "volume_cm3": volume_m3 * 1e6,
        "voxel_size": voxel_size,
        "depth_tolerance": depth_tolerance,
        "min_views_checked": min_views_checked,
        "num_voxels_total": num_total,
        "num_voxels_kept": num_kept,
        "expected_bias": "visual hull usually overestimates concave objects",
        "outputs": {
            "carved_voxels_ply": str(out / "carved_voxels.ply"),
            "carved_voxels_npz": str(out / "carved_voxels.npz"),
        },
    }
    report.update(gt_comparison_fields(scan, volume_m3))
    write_report(out / "report.json", report)
    return report

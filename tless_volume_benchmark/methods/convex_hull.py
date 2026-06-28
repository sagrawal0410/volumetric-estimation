"""Convex hull volume from fused back-projected depth points."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from tless_volume_benchmark.geometry import backproject_masked_depth_to_object
from tless_volume_benchmark.scan_io import (
    PreparedScan,
    gt_comparison_fields,
    load_prepared_scan,
    resolve_output_dir,
    write_report,
)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return points[np.sort(idx)]


def _remove_statistical_outliers(points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    if points.shape[0] <= nb_neighbors:
        return points
    # NumPy fallback is the default: broken scipy wheels (x86_64 on arm64) can segfault in cKDTree.
    if os.environ.get("TLESS_USE_SCIPY", "0") == "1":
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(points)
            dists, _ = tree.query(points, k=min(nb_neighbors + 1, points.shape[0]))
            mean_dist = dists[:, 1:].mean(axis=1)
            thresh = float(mean_dist.mean() + std_ratio * mean_dist.std())
            return points[mean_dist <= thresh]
        except Exception:
            pass

    k = min(nb_neighbors, max(1, points.shape[0] - 1))
    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    mean_dist = np.partition(dists, k, axis=1)[:, :k].mean(axis=1)
    thresh = float(mean_dist.mean() + std_ratio * mean_dist.std())
    return points[mean_dist <= thresh]


def _largest_cluster(points: np.ndarray, eps: float, min_samples: int = 10) -> np.ndarray:
    if points.shape[0] < min_samples:
        return points
    if os.environ.get("TLESS_USE_SKLEARN", "0") != "1":
        return points
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return points
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    valid = labels[labels >= 0]
    if valid.size == 0:
        return points
    unique, counts = np.unique(valid, return_counts=True)
    best = unique[int(np.argmax(counts))]
    return points[labels == best]


def fuse_points(scan: PreparedScan, depth_min: float = 0.05, depth_max: float = 5.0) -> np.ndarray:
    chunks = []
    for frame in scan.frames:
        pts = backproject_masked_depth_to_object(
            frame.depth_m, frame.mask, frame.K, frame.T_cam_to_object,
            depth_min=depth_min, depth_max=depth_max,
        )
        if pts.size:
            chunks.append(pts)
    if not chunks:
        raise ValueError("No valid depth points recovered from any frame")
    return np.vstack(chunks)


def estimate_convex_hull(
    scan_dir: str | Path,
    voxel_downsample: float = 0.0015,
    output_dir: str | Path | None = None,
    depth_min: float = 0.05,
    depth_max: float = 5.0,
) -> dict[str, Any]:
    scan = load_prepared_scan(scan_dir)
    out = resolve_output_dir(scan.scan_dir, "convex_hull", Path(output_dir) if output_dir else None)
    out.mkdir(parents=True, exist_ok=True)

    points = fuse_points(scan, depth_min=depth_min, depth_max=depth_max)
    points = _voxel_downsample(points, voxel_downsample)
    points = _remove_statistical_outliers(points)
    points = _largest_cluster(points, eps=3.0 * voxel_downsample)
    if points.shape[0] < 4:
        raise ValueError(f"Need at least 4 points for convex hull, got {points.shape[0]}")

    hull = trimesh.convex.convex_hull(points)
    volume_m3 = abs(float(hull.volume))

    trimesh.PointCloud(points).export(out / "fused_pointcloud.ply")
    hull.export(out / "hull_mesh.ply")

    report: dict[str, Any] = {
        "method": "convex_hull",
        "scan_dir": str(scan.scan_dir),
        "volume_m3": volume_m3,
        "volume_cm3": volume_m3 * 1e6,
        "num_points": int(points.shape[0]),
        "voxel_downsample": voxel_downsample,
        "expected_bias": "usually overestimates non-convex objects",
        "outputs": {
            "fused_pointcloud": str(out / "fused_pointcloud.ply"),
            "hull_mesh": str(out / "hull_mesh.ply"),
        },
    }
    report.update(gt_comparison_fields(scan, volume_m3))
    write_report(out / "report.json", report)
    return report

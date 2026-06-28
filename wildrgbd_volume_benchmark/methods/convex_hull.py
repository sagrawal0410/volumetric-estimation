"""Convex hull volume from sparse-view fused depth."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from wildrgbd_volume_benchmark.geometry import backproject_depth_to_object
from wildrgbd_volume_benchmark.scan_io import (
    load_prepared_scene,
    pseudo_gt_comparison_fields,
    resolve_output_dir,
    write_report,
)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return points[np.sort(idx)]


def _remove_outliers(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 20:
        return points
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        dists, _ = tree.query(points, k=min(21, points.shape[0]))
        mean_dist = dists[:, 1:].mean(axis=1)
        return points[mean_dist <= mean_dist.mean() + 2.0 * mean_dist.std()]
    except Exception:
        return points


def _largest_cluster(points: np.ndarray, eps: float) -> np.ndarray:
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return points
    labels = DBSCAN(eps=eps, min_samples=10).fit_predict(points)
    valid = labels[labels >= 0]
    if valid.size == 0:
        return points
    unique, counts = np.unique(valid, return_counts=True)
    return points[labels == unique[int(np.argmax(counts))]]


def estimate_convex_hull_volume(
    prepared_scene_dir: str | Path,
    voxel_downsample: float = 0.0025,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    scene = load_prepared_scene(prepared_scene_dir)
    out = resolve_output_dir(scene.scene_dir, "convex_hull", Path(output_dir) if output_dir else None)
    out.mkdir(parents=True, exist_ok=True)

    chunks = []
    for frame in scene.frames:
        pts = backproject_depth_to_object(frame.depth_m, frame.mask, frame.K, frame.T_cam_to_object)
        if pts.size:
            chunks.append(pts)
    points = np.vstack(chunks)
    points = _voxel_downsample(points, voxel_downsample)
    points = _remove_outliers(points)
    points = _largest_cluster(points, 3.0 * voxel_downsample)

    hull = trimesh.convex.convex_hull(points)
    volume_m3 = abs(float(hull.volume))

    trimesh.PointCloud(points).export(out / "fused_sampled_pointcloud.ply")
    hull.export(out / "convex_hull_mesh.ply")

    report: dict[str, Any] = {
        "method": "convex_hull",
        "prepared_scene_dir": str(scene.scene_dir),
        "volume_m3": volume_m3,
        "volume_cm3": volume_m3 * 1e6,
        "expected_bias": "usually overestimates concave objects",
        "notes": "Compared against full-video pseudo-GT, not exact scalar GT",
        "status": "ok",
    }
    report.update(pseudo_gt_comparison_fields(scene, volume_m3))
    write_report(out / "report.json", report)
    return report

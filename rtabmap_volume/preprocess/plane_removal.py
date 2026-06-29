"""RANSAC plane segmentation and removal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from rtabmap_volume.config import PlaneRemovalConfig


@dataclass
class PlaneResult:
    plane_model: np.ndarray  # [a, b, c, d] ax+by+cz+d=0
    inliers: np.ndarray
    outliers: np.ndarray
    inlier_ratio: float
    support_mesh: o3d.geometry.TriangleMesh | None = None


def segment_plane(pcd: o3d.geometry.PointCloud, cfg: PlaneRemovalConfig, seed: int = 42) -> PlaneResult:
    if len(pcd.points) < cfg.ransac_n:
        empty = np.array([], dtype=int)
        return PlaneResult(np.zeros(4), empty, empty, 0.0)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=cfg.distance_threshold_m,
        ransac_n=cfg.ransac_n,
        num_iterations=cfg.num_iterations,
    )
    inliers = np.asarray(inliers, dtype=int)
    n_pts = len(pcd.points)
    all_idx = np.arange(n_pts)
    outlier_mask = np.ones(n_pts, dtype=bool)
    outlier_mask[inliers] = False
    outliers = all_idx[outlier_mask]
    ratio = len(inliers) / max(n_pts, 1)

    support = _plane_mesh(pcd, plane_model, inliers) if len(inliers) > 0 else None
    return PlaneResult(
        plane_model=np.asarray(plane_model),
        inliers=inliers,
        outliers=outliers,
        inlier_ratio=ratio,
        support_mesh=support,
    )


def _plane_mesh(pcd: o3d.geometry.PointCloud, plane: np.ndarray, inliers: np.ndarray) -> o3d.geometry.TriangleMesh:
    pts = np.asarray(pcd.points)[inliers]
    if len(pts) < 3:
        return o3d.geometry.TriangleMesh()
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    # Simple quad in XY of plane bbox
    a, b, c, d = plane
    normal = np.array([a, b, c])
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    # Build two tangent vectors
    arb = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, arb)
    t1 /= np.linalg.norm(t1) + 1e-12
    t2 = np.cross(normal, t1)
    center = pts.mean(axis=0)
    size = max(mx - mn)
    corners = [
        center + s1 * t1 * size / 2 + s2 * t2 * size / 2
        for s1 in (-1, 1)
        for s2 in (-1, 1)
    ]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    mesh.paint_uniform_color([0.2, 0.6, 0.2])
    return mesh


def remove_plane_points(pcd: o3d.geometry.PointCloud, plane_result: PlaneResult) -> o3d.geometry.PointCloud:
    if len(plane_result.outliers) == 0:
        return o3d.geometry.PointCloud()
    pts = np.asarray(pcd.points)[plane_result.outliers]
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts)
    if pcd.has_colors():
        cropped.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[plane_result.outliers])
    if pcd.has_normals():
        cropped.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[plane_result.outliers])
    return cropped


def filter_above_plane(
    pcd: o3d.geometry.PointCloud,
    plane_model: np.ndarray,
    threshold_m: float,
) -> o3d.geometry.PointCloud:
    a, b, c, d = plane_model
    pts = np.asarray(pcd.points)
    dist = (pts @ np.array([a, b, c]) + d) / (np.linalg.norm([a, b, c]) + 1e-12)
    mask = dist > threshold_m
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        cropped.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        cropped.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
    return cropped

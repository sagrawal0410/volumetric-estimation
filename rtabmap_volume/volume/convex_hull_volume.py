"""Convex hull volume — upper bound estimator."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import ConvexHull

from rtabmap_volume.volume.mesh_volume import VolumeEstimate, _liters


def compute_convex_hull_volume(
    pcd: o3d.geometry.PointCloud | None = None,
    mesh: trimesh.Trimesh | None = None,
    min_points: int = 4,
) -> VolumeEstimate:
    if mesh is not None and len(mesh.vertices) >= min_points:
        points = np.asarray(mesh.vertices)
    elif pcd is not None and len(pcd.points) >= min_points:
        points = np.asarray(pcd.points)
    else:
        return VolumeEstimate(
            "convex_hull_volume",
            None,
            None,
            False,
            ["Not enough points for convex hull"],
        )

    try:
        hull = ConvexHull(points)
        vol = float(hull.volume)
    except Exception as e:
        return VolumeEstimate("convex_hull_volume", None, None, False, [f"Convex hull failed: {e}"])

    return VolumeEstimate(
        "convex_hull_volume",
        vol,
        _liters(vol),
        False,
        ["Convex hull is an upper bound; often overestimates concave objects"],
        metadata={"is_upper_bound": True},
    )

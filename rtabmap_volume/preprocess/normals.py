"""Normal estimation for point clouds."""

from __future__ import annotations

import open3d as o3d


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    radius_m: float = 0.05,
    max_nn: int = 30,
    orient_toward_camera: bool = False,
) -> o3d.geometry.PointCloud:
    result = o3d.geometry.PointCloud(pcd)
    if len(result.points) == 0:
        return result
    result.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_m, max_nn=max_nn)
    )
    if orient_toward_camera:
        result.orient_normals_towards_camera_location(camera_location=[0.0, 0.0, 0.0])
    else:
        try:
            result.orient_normals_consistent_tangent_plane(k=min(15, max(3, len(result.points) // 100)))
        except RuntimeError:
            # Fallback when points are degenerate/coplanar
            result.orient_normals_towards_camera_location(camera_location=[0.0, 0.0, 1.0])
    return result

"""Plane removal and clustering tests."""

import numpy as np
import open3d as o3d

from rtabmap_volume.config import ClusteringConfig, PlaneRemovalConfig
from rtabmap_volume.preprocess.clustering import cluster_point_cloud, extract_cluster
from rtabmap_volume.preprocess.plane_removal import remove_plane_points, segment_plane


def _make_plane_and_cube():
    rng = np.random.default_rng(42)
    plane_pts = rng.uniform([-1, -1, 0], [1, 1, 0], (2000, 3))
    cube_pts = rng.uniform([0.2, 0.2, 0.05], [0.5, 0.5, 0.4], (800, 3))
    pts = np.vstack([plane_pts, cube_pts])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def test_plane_removal_keeps_cube_cluster():
    pcd = _make_plane_and_cube()
    plane = segment_plane(pcd, PlaneRemovalConfig(distance_threshold_m=0.02))
    assert plane.inlier_ratio > 0.3
    no_plane = remove_plane_points(pcd, plane)
    cluster = cluster_point_cloud(no_plane, ClusteringConfig(eps_m=0.08, min_points=20))
    segmented = extract_cluster(no_plane, cluster.labels, cluster.selected_cluster_id)
    pts = np.asarray(segmented.points)
    assert len(pts) > 100
    assert pts[:, 2].min() > 0.02

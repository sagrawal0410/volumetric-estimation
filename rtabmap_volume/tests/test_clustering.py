"""Tests for DBSCAN clustering."""

import numpy as np
import open3d as o3d

from rtabmap_volume.config import ClusteringConfig
from rtabmap_volume.preprocess.clustering import cluster_point_cloud


def test_two_clusters_selects_largest():
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0, 0], 0.05, (500, 3))
    b = rng.normal([1, 1, 1], 0.05, (100, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([a, b]))
    result = cluster_point_cloud(pcd, ClusteringConfig(eps_m=0.2, min_points=10))
    assert result.selected_cluster_id >= 0
    assert len(result.summary_df) >= 1

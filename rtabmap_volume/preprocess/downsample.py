"""Voxel downsampling."""

from __future__ import annotations

import open3d as o3d


def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size_m: float) -> o3d.geometry.PointCloud:
    if voxel_size_m <= 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size_m)

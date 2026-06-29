"""Point cloud denoising and downsampling."""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d

from rtabmap_volume.config import DenoiseConfig


def remove_nan_inf(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    valid = np.isfinite(pts).all(axis=1)
    cleaned = o3d.geometry.PointCloud()
    cleaned.points = o3d.utility.Vector3dVector(pts[valid])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        cleaned.colors = o3d.utility.Vector3dVector(colors[valid])
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        cleaned.normals = o3d.utility.Vector3dVector(normals[valid])
    return cleaned


def denoise_point_cloud(pcd: o3d.geometry.PointCloud, cfg: DenoiseConfig) -> o3d.geometry.PointCloud:
    result = remove_nan_inf(copy.deepcopy(pcd))
    n = len(result.points)

    if cfg.voxel_downsample_m and cfg.voxel_downsample_m > 0 and n > 100:
        result = result.voxel_down_sample(cfg.voxel_downsample_m)

    if cfg.enable_statistical and len(result.points) > max(cfg.statistical_nb_neighbors * 3, 100):
        result, _ = result.remove_statistical_outlier(
            nb_neighbors=cfg.statistical_nb_neighbors,
            std_ratio=cfg.statistical_std_ratio,
        )

    if cfg.enable_radius and len(result.points) > max(cfg.radius_nb_points * 3, 100):
        result, _ = result.remove_radius_outlier(
            nb_points=cfg.radius_nb_points,
            radius=cfg.radius_search_m,
        )

    return result

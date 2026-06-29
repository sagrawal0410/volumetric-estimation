"""Object/pile segmentation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import open3d as o3d
import trimesh

from rtabmap_volume.config import ClusteringConfig, PlaneRemovalConfig, PipelineConfig
from rtabmap_volume.preprocess.clustering import cluster_point_cloud, extract_cluster, save_cluster_summary
from rtabmap_volume.preprocess.crop import ROIBox, apply_roi, load_roi_json
from rtabmap_volume.preprocess.plane_removal import filter_above_plane, remove_plane_points, segment_plane
from rtabmap_volume.io.load_geometry import mesh_to_dense_point_cloud


@dataclass
class SegmentationResult:
    mesh: trimesh.Trimesh | None
    point_cloud: o3d.geometry.PointCloud | None
    diagnostics: dict[str, Any]
    warnings: list[str]


def segment_geometry(
    mesh: trimesh.Trimesh | None,
    pcd: o3d.geometry.PointCloud | None,
    config: PipelineConfig,
    roi_json: str | None = None,
    seed: int = 42,
) -> SegmentationResult:
    mode = config.segmentation.mode
    diagnostics: dict[str, Any] = {"mode": mode}
    warnings: list[str] = []

    if mesh is not None and pcd is None:
        pcd = mesh_to_dense_point_cloud(mesh)

    if mode == "none":
        return SegmentationResult(mesh, pcd, diagnostics, warnings)

    if mode in ("manual_aabb", "manual_obb"):
        if not roi_json:
            warnings.append(f"Segmentation mode {mode} requires --roi_json")
            return SegmentationResult(mesh, pcd, diagnostics, warnings)
        roi = load_roi_json(roi_json)
        new_mesh, new_pcd = apply_roi(mesh, pcd, roi)
        diagnostics["roi"] = roi.data
        return SegmentationResult(new_mesh, new_pcd, diagnostics, warnings)

    if mode == "plane_then_cluster":
        if pcd is None:
            warnings.append("plane_then_cluster requires point cloud")
            return SegmentationResult(mesh, pcd, diagnostics, warnings)
        plane_cfg = config.plane_removal
        plane = segment_plane(pcd, plane_cfg, seed=seed)
        diagnostics["plane_model"] = plane.plane_model.tolist()
        diagnostics["plane_inlier_ratio"] = plane.inlier_ratio
        if plane.inlier_ratio < plane_cfg.min_plane_inlier_ratio:
            warnings.append("Dominant plane inlier ratio below threshold; plane removal may be unreliable")
        no_plane = remove_plane_points(pcd, plane)
        cluster_cfg = config.clustering
        cluster = cluster_point_cloud(
            no_plane,
            cluster_cfg,
            cluster_id=config.segmentation.cluster_id,
            seed=seed,
        )
        diagnostics["cluster_summary"] = cluster.summary_df.to_dict(orient="records")
        diagnostics["selected_cluster_id"] = cluster.selected_cluster_id
        segmented = extract_cluster(no_plane, cluster.labels, cluster.selected_cluster_id)
        return SegmentationResult(None, segmented, diagnostics, warnings)

    if mode == "height_above_plane":
        if pcd is None:
            warnings.append("height_above_plane requires point cloud")
            return SegmentationResult(mesh, pcd, diagnostics, warnings)
        plane = segment_plane(pcd, config.plane_removal, seed=seed)
        diagnostics["plane_model"] = plane.plane_model.tolist()
        threshold = config.segmentation.height_above_plane_threshold_m
        segmented = filter_above_plane(pcd, plane.plane_model, threshold)
        diagnostics["height_threshold_m"] = threshold
        return SegmentationResult(None, segmented, diagnostics, warnings)

    warnings.append(f"Unknown segmentation mode: {mode}")
    return SegmentationResult(mesh, pcd, diagnostics, warnings)

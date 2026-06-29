"""Mesh reconstruction from point clouds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import trimesh

from rtabmap_volume.config import AlphaShapeConfig, BallPivotingConfig, PoissonConfig
from rtabmap_volume.io.load_geometry import open3d_to_trimesh
from rtabmap_volume.mesh.mesh_cleaning import clean_mesh
from rtabmap_volume.config import MeshCleaningConfig
from rtabmap_volume.preprocess.normals import estimate_normals


@dataclass
class ReconstructedMesh:
    name: str
    mesh: trimesh.Trimesh | None
    mesh_o3d: o3d.geometry.TriangleMesh | None
    warnings: list[str]


def _o3d_to_trimesh(mesh_o3d: o3d.geometry.TriangleMesh) -> trimesh.Trimesh | None:
    if mesh_o3d.is_empty() or len(mesh_o3d.triangles) == 0:
        return None
    return open3d_to_trimesh(mesh_o3d)


def reconstruct_poisson(pcd: o3d.geometry.PointCloud, cfg: PoissonConfig) -> ReconstructedMesh:
    warnings: list[str] = []
    if len(pcd.points) < 100:
        return ReconstructedMesh("poisson", None, None, ["Too few points for Poisson"])

    pcd_n = estimate_normals(pcd)
    try:
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_n,
            depth=cfg.depth,
            width=cfg.width,
            scale=cfg.scale,
            linear_fit=cfg.linear_fit,
        )
    except Exception as e:
        return ReconstructedMesh("poisson", None, None, [f"Poisson failed: {e}"])

    if len(densities) > 0:
        dens = np.asarray(densities)
        thresh = np.quantile(dens, cfg.density_quantile)
        verts_to_remove = dens < thresh
        mesh_o3d.remove_vertices_by_mask(verts_to_remove)

    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_non_manifold_edges()

    mesh = _o3d_to_trimesh(mesh_o3d)
    if mesh is None:
        warnings.append("Poisson produced empty mesh")
    return ReconstructedMesh("poisson", mesh, mesh_o3d, warnings)


def reconstruct_ball_pivoting(pcd: o3d.geometry.PointCloud, cfg: BallPivotingConfig) -> ReconstructedMesh:
    warnings: list[str] = []
    if len(pcd.points) < 50:
        return ReconstructedMesh("bpa", None, None, ["Too few points for BPA"])

    pcd_n = estimate_normals(pcd)
    pts = np.asarray(pcd_n.points)
    if len(pts) < 2:
        return ReconstructedMesh("bpa", None, None, ["Too few points"])

    tree = o3d.geometry.KDTreeFlann(pcd_n)
    dists = []
    for i in range(min(100, len(pts))):
        _, _, d2 = tree.search_knn_vector_3d(pts[i], 2)
        if len(d2) >= 2:
            dists.append(np.sqrt(d2[1]))
    if not dists:
        return ReconstructedMesh("bpa", None, None, ["Could not estimate neighbor distance"])
    avg_d = float(np.median(dists))
    radii = [avg_d * m for m in cfg.radius_multipliers]

    try:
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd_n, o3d.utility.DoubleVector(radii)
        )
    except Exception as e:
        return ReconstructedMesh("bpa", None, None, [f"BPA failed: {e}"])

    mesh = _o3d_to_trimesh(mesh_o3d)
    if mesh is None or not mesh.is_watertight:
        warnings.append("Ball pivoting mesh often not watertight")
    return ReconstructedMesh("bpa", mesh, mesh_o3d, warnings)


def reconstruct_alpha_shape(pcd: o3d.geometry.PointCloud, cfg: AlphaShapeConfig) -> ReconstructedMesh:
    warnings: list[str] = []
    if len(pcd.points) < 20:
        return ReconstructedMesh("alpha_shape", None, None, ["Too few points for alpha shape"])

    best_mesh: trimesh.Trimesh | None = None
    best_o3d: o3d.geometry.TriangleMesh | None = None
    best_score = -1.0

    for alpha in cfg.alpha_values:
        try:
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        except Exception:
            continue
        if mesh_o3d.is_empty():
            continue
        mesh = _o3d_to_trimesh(mesh_o3d)
        if mesh is None:
            continue
        wt = 1.0 if mesh.is_watertight else 0.3
        bbox_vol = float(np.prod(mesh.bounds[1] - mesh.bounds[0])) + 1e-12
        vol_ratio = min(mesh.volume / bbox_vol, 1.0) if mesh.is_watertight else 0.5
        score = wt * vol_ratio
        if score > best_score:
            best_score = score
            best_mesh = mesh
            best_o3d = mesh_o3d

    if best_mesh is None:
        warnings.append("No suitable alpha shape found")
    return ReconstructedMesh("alpha_shape", best_mesh, best_o3d, warnings)

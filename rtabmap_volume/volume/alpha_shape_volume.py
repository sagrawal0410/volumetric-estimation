"""Alpha shape volume from reconstructed mesh."""

from __future__ import annotations

import trimesh

from rtabmap_volume.config import AlphaShapeConfig
from rtabmap_volume.mesh.mesh_from_pointcloud import reconstruct_alpha_shape
from rtabmap_volume.volume.mesh_volume import VolumeEstimate, compute_mesh_volume
import open3d as o3d


def compute_alpha_shape_volume(
    pcd: o3d.geometry.PointCloud,
    cfg: AlphaShapeConfig | None = None,
) -> tuple[VolumeEstimate, trimesh.Trimesh | None, dict[float, float | None]]:
    cfg = cfg or AlphaShapeConfig()
    alpha_volumes: dict[float, float | None] = {}
    unstable = False

    for alpha in cfg.alpha_values:
        try:
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            if mesh_o3d.is_empty():
                alpha_volumes[alpha] = None
                continue
            from rtabmap_volume.io.load_geometry import open3d_to_trimesh
            mesh = open3d_to_trimesh(mesh_o3d)
            est = compute_mesh_volume(mesh, "alpha_shape_volume")
            alpha_volumes[alpha] = est.value_m3
        except Exception:
            alpha_volumes[alpha] = None

    valid = [v for v in alpha_volumes.values() if v is not None and v > 0]
    if len(valid) >= 2:
        spread = (max(valid) - min(valid)) / (sum(valid) / len(valid))
        unstable = spread > 0.3

    recon = reconstruct_alpha_shape(pcd, cfg)
    est = compute_mesh_volume(recon.mesh, "alpha_shape_volume")
    if unstable:
        est.reliable = False
        est.warnings.append("Alpha shape volume unstable across alpha values")
    est.metadata = {"alpha_volumes": {str(k): v for k, v in alpha_volumes.items()}}
    return est, recon.mesh, alpha_volumes

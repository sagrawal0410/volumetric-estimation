"""Voxel occupancy volume estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import trimesh

from rtabmap_volume.config import VoxelConfig
from rtabmap_volume.volume.mesh_volume import VolumeEstimate, _liters


@dataclass
class VoxelGridResult:
    occupied: np.ndarray
    voxel_size: float
    origin: np.ndarray
    volume_m3: float


def _voxelize_points(points: np.ndarray, voxel_size: float, _depth: int = 0) -> VoxelGridResult:
    if len(points) == 0:
        return VoxelGridResult(np.zeros((1, 1, 1), dtype=bool), voxel_size, np.zeros(3), 0.0)
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    dims = np.ceil((mx - mn) / voxel_size).astype(int) + 1
    dims = np.maximum(dims, 1)
    grid = np.zeros(dims, dtype=bool)

    # Mark surface voxels from points
    idx = np.floor((points - mn) / voxel_size).astype(int)
    idx = np.clip(idx, 0, dims - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # Surface-only voxelization severely underestimates volume for point clouds.
    # Fill interior using convex hull inclusion test when occupancy is sparse.
    surface_vol = float(grid.sum()) * (voxel_size ** 3)
    bbox_vol = float(np.prod(mx - mn))
    max_voxels = 500_000
    total_voxels = int(np.prod(dims))
    if surface_vol < 0.15 * bbox_vol and len(points) >= 4 and total_voxels <= max_voxels:
        try:
            from scipy.spatial import Delaunay, ConvexHull

            hull = ConvexHull(points)
            delaunay = Delaunay(points[hull.vertices])
            xs = mn[0] + (np.arange(dims[0]) + 0.5) * voxel_size
            ys = mn[1] + (np.arange(dims[1]) + 0.5) * voxel_size
            zs = mn[2] + (np.arange(dims[2]) + 0.5) * voxel_size
            xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
            centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
            inside = delaunay.find_simplex(centers) >= 0
            grid = inside.reshape(dims)
        except Exception:
            pass
    elif surface_vol < 0.15 * bbox_vol and total_voxels > max_voxels and _depth < 3:
        coarse = _voxelize_points(points, voxel_size * 2, _depth=_depth + 1)
        vol = coarse.volume_m3 / 8.0
        return VoxelGridResult(coarse.occupied, voxel_size, mn, vol)

    vol = float(grid.sum()) * (voxel_size ** 3)
    return VoxelGridResult(grid, voxel_size, mn, vol)


def _voxelize_mesh(mesh: trimesh.Trimesh, voxel_size: float) -> VoxelGridResult:
    try:
        voxels = mesh.voxelized(pitch=voxel_size)
        filled = voxels.fill().matrix if hasattr(voxels.fill(), "matrix") else voxels.matrix
        vol = float(filled.sum()) * (voxel_size ** 3)
        return VoxelGridResult(filled, voxel_size, mesh.bounds[0], vol)
    except Exception:
        return _voxelize_points(np.asarray(mesh.vertices), voxel_size)


def compute_voxel_volumes(
    pcd: o3d.geometry.PointCloud | None = None,
    mesh: trimesh.Trimesh | None = None,
    cfg: VoxelConfig | None = None,
) -> tuple[VolumeEstimate, dict[float, VoxelGridResult]]:
    cfg = cfg or VoxelConfig()
    results: dict[float, VoxelGridResult] = {}
    volumes: list[float] = []

    for vs in sorted(cfg.voxel_sizes_m):
        if mesh is not None and len(mesh.faces) > 0:
            vgr = _voxelize_mesh(mesh, vs)
        elif pcd is not None and len(pcd.points) > 0:
            vgr = _voxelize_points(np.asarray(pcd.points), vs)
        else:
            continue
        results[vs] = vgr
        volumes.append(vgr.volume_m3)

    if not volumes:
        return VolumeEstimate("voxel_occupancy_volume", None, None, False, ["No geometry to voxelize"]), results

    # Use finest resolution as primary, check convergence
    finest = min(results.keys())
    primary = results[finest].volume_m3
    warnings: list[str] = []
    spread = 0.0
    if len(volumes) >= 2:
        spread = (max(volumes) - min(volumes)) / (np.median(volumes) + 1e-12)
        if spread > 0.15:
            warnings.append(f"Voxel volume varies {spread*100:.1f}% across resolutions")
    reliable = spread < 0.1 if len(volumes) >= 2 else False

    return (
        VolumeEstimate(
            "voxel_occupancy_volume",
            primary,
            _liters(primary),
            reliable,
            warnings,
            metadata={"by_voxel_size": {str(k): v.volume_m3 for k, v in results.items()}},
        ),
        results,
    )

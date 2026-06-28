"""Mesh volume computation with watertight checks and fallbacks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


@dataclass
class VolumeReport:
    mesh_watertight: bool
    volume_m3: float
    volume_liters: float
    fallback_method: str | None
    surface_area_m2: float
    bbox_volume_m3: float
    num_vertices: int
    num_faces: int
    volume_status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_mesh_volume_report(mesh: trimesh.Trimesh, voxel_size_m: float = 0.003) -> VolumeReport:
    mesh = mesh.copy()
    watertight = bool(mesh.is_watertight)
    fallback = None
    status = "ok"

    if watertight:
        volume_m3 = float(abs(mesh.volume))
    else:
        status = "not_watertight_fallback"
        try:
            vox = mesh.voxelized(pitch=voxel_size_m)
            volume_m3 = float(vox.filled_count * (voxel_size_m**3))
            fallback = "voxelized"
        except Exception:  # noqa: BLE001
            volume_m3 = float(abs(mesh.convex_hull.volume))
            fallback = "convex_hull"

    bbox = mesh.bounds
    bbox_vol = float(np.prod(bbox[1] - bbox[0]))
    area = float(mesh.area) if mesh.faces.size else 0.0

    return VolumeReport(
        mesh_watertight=watertight,
        volume_m3=volume_m3,
        volume_liters=volume_m3 * 1000.0,
        fallback_method=fallback,
        surface_area_m2=area,
        bbox_volume_m3=bbox_vol,
        num_vertices=int(len(mesh.vertices)),
        num_faces=int(len(mesh.faces)),
        volume_status=status,
    )


def mesh_volume_m3(mesh: trimesh.Trimesh) -> float:
    return compute_mesh_volume_report(mesh).volume_m3


def load_mesh_volume_report(path: Path, voxel_size_m: float = 0.003) -> VolumeReport:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh at {path}")
    return compute_mesh_volume_report(mesh, voxel_size_m=voxel_size_m)


def voxelize_mesh(mesh: trimesh.Trimesh, voxel_size_m: float) -> np.ndarray:
    voxels = mesh.voxelized(pitch=voxel_size_m)
    return np.asarray(voxels.matrix, dtype=bool)


def union_voxel_grids(grids: list[np.ndarray]) -> np.ndarray:
    if not grids:
        raise ValueError("No voxel grids to union")
    out = grids[0].copy()
    for g in grids[1:]:
        if g.shape != out.shape:
            raise ValueError("Voxel grid shapes must match for union")
        out |= g
    return out

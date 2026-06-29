"""Mesh cleaning utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from rtabmap_volume.config import MeshCleaningConfig


@dataclass
class MeshQualityStats:
    vertex_count: int
    face_count: int
    connected_components: int
    is_watertight: bool
    euler_number: int | None
    volume_m3: float | None


def mesh_quality_stats(mesh: trimesh.Trimesh) -> MeshQualityStats:
    vol = float(mesh.volume) if mesh.is_watertight and mesh.is_volume else None
    euler = None
    try:
        euler = int(mesh.euler_number)
    except Exception:
        pass
    n_comp = len(mesh.split(only_watertight=False))
    return MeshQualityStats(
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
        connected_components=n_comp,
        is_watertight=bool(mesh.is_watertight),
        euler_number=euler,
        volume_m3=vol,
    )


def clean_mesh(mesh: trimesh.Trimesh, cfg: MeshCleaningConfig) -> tuple[trimesh.Trimesh, MeshQualityStats, MeshQualityStats]:
    before = mesh_quality_stats(mesh)
    cleaned = mesh.copy()

    cleaned.update_faces(cleaned.unique_faces())
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.remove_unreferenced_vertices()

    if cfg.merge_vertices_tolerance_m > 0:
        cleaned.merge_vertices(merge_tex=True, merge_norm=True)

    if cfg.keep_largest_component:
        parts = cleaned.split(only_watertight=False)
        if parts:
            cleaned = max(parts, key=lambda m: len(m.faces))

    if cfg.fill_small_holes:
        try:
            trimesh.repair.fill_holes(cleaned)
        except Exception:
            pass

    try:
        trimesh.repair.fix_normals(cleaned)
        trimesh.repair.fix_winding(cleaned)
    except Exception:
        pass

    after = mesh_quality_stats(cleaned)
    return cleaned, before, after

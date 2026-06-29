"""Mesh repair using trimesh and optional pymeshlab."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import trimesh

from rtabmap_volume.config import MeshRepairConfig
from rtabmap_volume.mesh.mesh_cleaning import MeshQualityStats, mesh_quality_stats


@dataclass
class RepairReport:
    before: MeshQualityStats
    after: MeshQualityStats
    changes: dict[str, int | float | bool | None]
    pymeshlab_used: bool


def _repair_trimesh(mesh: trimesh.Trimesh, cfg: MeshRepairConfig) -> trimesh.Trimesh:
    repaired = mesh.copy()
    if cfg.fix_normals:
        trimesh.repair.fix_normals(repaired)
    if cfg.fix_winding:
        trimesh.repair.fix_winding(repaired)
    if cfg.fix_inversion:
        trimesh.repair.fix_inversion(repaired)
    if cfg.fill_holes:
        trimesh.repair.fill_holes(repaired)
    repaired.update_faces(repaired.unique_faces())
    repaired.update_faces(repaired.nondegenerate_faces())
    repaired.remove_unreferenced_vertices()
    return repaired


def _repair_pymeshlab(mesh: trimesh.Trimesh) -> trimesh.Trimesh | None:
    try:
        import pymeshlab  # type: ignore
    except ImportError:
        return None

    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)
    ms.add_mesh(m)
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=10)
    ms.meshing_close_holes(maxholesize=100)
    ms.meshing_repair_non_manifold_edges()
    out = ms.current_mesh()
    return trimesh.Trimesh(vertices=out.vertex_matrix(), faces=out.face_matrix(), process=False)


def repair_mesh(mesh: trimesh.Trimesh, cfg: MeshRepairConfig) -> tuple[trimesh.Trimesh, RepairReport]:
    before = mesh_quality_stats(mesh)
    repaired = _repair_trimesh(mesh, cfg)
    pymeshlab_used = False

    if cfg.use_pymeshlab:
        pm = _repair_pymeshlab(repaired)
        if pm is not None:
            repaired = pm
            pymeshlab_used = True

    after = mesh_quality_stats(repaired)
    changes = {
        "vertex_delta": after.vertex_count - before.vertex_count,
        "face_delta": after.face_count - before.face_count,
        "components_delta": after.connected_components - before.connected_components,
        "watertight_before": before.is_watertight,
        "watertight_after": after.is_watertight,
        "volume_before": before.volume_m3,
        "volume_after": after.volume_m3,
    }
    report = RepairReport(before=before, after=after, changes=changes, pymeshlab_used=pymeshlab_used)
    return repaired, report

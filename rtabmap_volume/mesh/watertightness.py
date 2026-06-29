"""Mesh watertightness assessment."""

from __future__ import annotations

from dataclasses import dataclass

import trimesh


@dataclass
class WatertightnessReport:
    is_watertight: bool
    is_volume: bool
    boundary_edge_count: int | None
    euler_number: int | None
    closeable: bool
    warnings: list[str]


def assess_watertightness(mesh: trimesh.Trimesh) -> WatertightnessReport:
    warnings: list[str] = []
    boundary_edges = None
    try:
        boundary_edges = len(mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)])
    except Exception:
        pass

    euler = None
    try:
        euler = int(mesh.euler_number)
    except Exception:
        pass

    is_wt = bool(mesh.is_watertight)
    is_vol = bool(mesh.is_volume)

    closeable = is_wt
    if not is_wt:
        warnings.append("Mesh is not watertight; direct mesh volume is not reliable")
        if boundary_edges and boundary_edges < len(mesh.faces) * 0.1:
            closeable = True
            warnings.append("Mesh has relatively few boundary edges; repair may produce usable volume")

    return WatertightnessReport(
        is_watertight=is_wt,
        is_volume=is_vol,
        boundary_edge_count=boundary_edges,
        euler_number=euler,
        closeable=closeable,
        warnings=warnings,
    )

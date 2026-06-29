"""Mesh quality metrics."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import trimesh

from rtabmap_volume.mesh.mesh_cleaning import MeshQualityStats, mesh_quality_stats
from rtabmap_volume.mesh.watertightness import WatertightnessReport, assess_watertightness


@dataclass
class FullMeshQuality:
    stats: MeshQualityStats
    watertightness: WatertightnessReport

    def to_dict(self) -> dict:
        return {
            "stats": asdict(self.stats),
            "watertightness": asdict(self.watertightness),
        }


def evaluate_mesh_quality(mesh: trimesh.Trimesh) -> FullMeshQuality:
    return FullMeshQuality(stats=mesh_quality_stats(mesh), watertightness=assess_watertightness(mesh))

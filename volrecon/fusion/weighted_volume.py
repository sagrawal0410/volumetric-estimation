"""Volume computation for weighted TSDF and occupancy outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from volrecon.geometry.mesh_volume import VolumeReport, compute_mesh_volume_report


@dataclass
class WeightedVolumeReport:
    mesh_volume: VolumeReport
    occupancy_volume_m3: float | None
    tsdf_voxel_volume_m3: float | None
    voxel_size_m: float

    def to_dict(self) -> dict[str, Any]:
        d = {
            "mesh": self.mesh_volume.to_dict(),
            "occupancy_volume_m3": self.occupancy_volume_m3,
            "tsdf_voxel_volume_m3": self.tsdf_voxel_volume_m3,
            "voxel_size_m": self.voxel_size_m,
        }
        return d


def volume_from_occupancy(prob_occ: np.ndarray, voxel_size_m: float, threshold: float = 0.5) -> float:
    return float((prob_occ > threshold).sum()) * (voxel_size_m**3)


def volume_from_tsdf_weight(tsdf: np.ndarray, weight: np.ndarray, min_weight: float, voxel_size_m: float) -> float:
    mask = weight >= min_weight
    return float(mask.sum()) * (voxel_size_m**3)


def compute_weighted_volumes(
    mesh_path: Path,
    voxel_size_m: float,
    occupancy_path: Path | None = None,
    occupancy_threshold: float = 0.5,
    tsdf_path: Path | None = None,
    weight_path: Path | None = None,
    min_weight: float = 2.0,
) -> WeightedVolumeReport:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    mesh_rep = compute_mesh_volume_report(mesh, voxel_size_m=voxel_size_m)

    occ_vol = None
    if occupancy_path and occupancy_path.exists():
        data = np.load(occupancy_path)
        prob = data["prob_occ"] if "prob_occ" in data else data["log_odds"]
        if "log_odds" in data and "prob_occ" not in data:
            lo = prob
            prob = 1.0 - 1.0 / (1.0 + np.exp(lo))
        occ_vol = volume_from_occupancy(prob, voxel_size_m, occupancy_threshold)

    tsdf_vol = None
    if tsdf_path and weight_path and tsdf_path.exists() and weight_path.exists():
        tsdf = np.load(tsdf_path)
        weight = np.load(weight_path)
        tsdf_vol = volume_from_tsdf_weight(tsdf, weight, min_weight, voxel_size_m)

    return WeightedVolumeReport(
        mesh_volume=mesh_rep,
        occupancy_volume_m3=occ_vol,
        tsdf_voxel_volume_m3=tsdf_vol,
        voxel_size_m=voxel_size_m,
    )

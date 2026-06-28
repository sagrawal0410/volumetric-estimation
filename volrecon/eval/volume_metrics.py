"""Volume comparison metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from volrecon.geometry.mesh_volume import VolumeReport, compute_mesh_volume_report
import trimesh


@dataclass
class VolumeComparison:
    predicted_volume_m3: float
    gt_volume_m3: float
    abs_volume_error_m3: float
    rel_volume_error_percent: float
    abs_volume_error_liters: float
    predicted_watertight: bool
    gt_watertight: bool
    gt_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_volumes(
    pred_mesh: trimesh.Trimesh,
    gt_mesh: trimesh.Trimesh,
    gt_source: str = "gt_mesh",
    voxel_size_m: float = 0.003,
) -> tuple[VolumeComparison, VolumeReport, VolumeReport]:
    pred_rep = compute_mesh_volume_report(pred_mesh, voxel_size_m=voxel_size_m)
    gt_rep = compute_mesh_volume_report(gt_mesh, voxel_size_m=voxel_size_m)
    abs_err = abs(pred_rep.volume_m3 - gt_rep.volume_m3)
    rel_err = 100.0 * abs_err / max(gt_rep.volume_m3, 1e-9)
    cmp = VolumeComparison(
        predicted_volume_m3=pred_rep.volume_m3,
        gt_volume_m3=gt_rep.volume_m3,
        abs_volume_error_m3=abs_err,
        rel_volume_error_percent=rel_err,
        abs_volume_error_liters=abs_err * 1000.0,
        predicted_watertight=pred_rep.mesh_watertight,
        gt_watertight=gt_rep.mesh_watertight,
        gt_source=gt_source,
    )
    return cmp, pred_rep, gt_rep

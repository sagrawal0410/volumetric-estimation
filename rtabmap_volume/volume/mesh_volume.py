"""Direct mesh volume estimation."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import trimesh

from rtabmap_volume.mesh.watertightness import assess_watertightness


@dataclass
class VolumeEstimate:
    name: str
    value_m3: float | None
    value_liters: float | None
    reliable: bool
    warnings: list[str]
    metadata: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _liters(m3: float | None) -> float | None:
    return m3 * 1000.0 if m3 is not None else None


def _bbox_sanity(mesh: trimesh.Trimesh, volume: float) -> bool:
    dims = mesh.bounds[1] - mesh.bounds[0]
    bbox_vol = float(np.prod(dims))
    if bbox_vol <= 0:
        return False
    ratio = volume / bbox_vol
    return 0.0 < ratio <= 1.05


def compute_mesh_volume(mesh: trimesh.Trimesh | None, name: str = "direct_mesh_volume") -> VolumeEstimate:
    warnings: list[str] = []
    if mesh is None or len(mesh.faces) == 0:
        return VolumeEstimate(name, None, None, False, ["Empty mesh"])

    wt = assess_watertightness(mesh)
    warnings.extend(wt.warnings)

    if not mesh.is_watertight:
        try:
            vol = float(mesh.volume)
        except Exception:
            vol = 0.0
        if vol > 0 and _bbox_sanity(mesh, vol):
            warnings.append("Mesh not watertight — volume reported as model-based estimate only")
            return VolumeEstimate(name, vol, _liters(vol), False, warnings)
        return VolumeEstimate(name, None, None, False, warnings + ["Mesh not watertight"])

    try:
        vol = float(mesh.volume)
    except Exception as e:
        return VolumeEstimate(name, None, None, False, warnings + [f"Volume computation failed: {e}"])

    if vol <= 0:
        return VolumeEstimate(name, None, None, False, warnings + ["Non-positive volume"])

    if not _bbox_sanity(mesh, vol):
        warnings.append("Volume exceeds bounding box sanity check")
        return VolumeEstimate(name, vol, _liters(vol), False, warnings)

    if not mesh.is_volume:
        warnings.append("Mesh winding may be inconsistent")
        return VolumeEstimate(name, vol, _liters(vol), False, warnings)

    return VolumeEstimate(name, vol, _liters(vol), True, warnings)

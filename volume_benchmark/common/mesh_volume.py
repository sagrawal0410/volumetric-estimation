"""Ground-truth mesh loading and volume computation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh

SourceUnits = Literal["m", "mm", "auto"]
GtType = Literal["mesh_watertight", "mesh_repaired", "full_reconstruction_pseudo_gt"]


def _infer_units_from_extent(mesh: trimesh.Trimesh) -> str:
    extent = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
    # Typical object sizes: <5 m in meters, >50 mm if stored in millimeters.
    return "mm" if extent > 50.0 else "m"


def load_mesh_as_meters(path: str | Path, source_units: SourceUnits = "auto") -> trimesh.Trimesh:
    """Load a mesh and ensure vertex coordinates are in meters."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = trimesh.load_mesh(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh)}")

    units = _infer_units_from_extent(mesh) if source_units == "auto" else source_units
    if units not in ("m", "mm"):
        raise ValueError(f"source_units must be 'm', 'mm', or 'auto', got {source_units!r}")

    if units == "mm":
        mesh = mesh.copy()
        mesh.apply_scale(0.001)

    return mesh


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove degenerate faces/vertices and merge close vertices."""
    cleaned = mesh.copy()
    cleaned.remove_infinite_values()
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.update_faces(cleaned.unique_faces())
    cleaned.remove_unreferenced_vertices()
    cleaned.merge_vertices()
    return cleaned


def compute_mesh_volume_m3(mesh: trimesh.Trimesh, repair: bool = False) -> tuple[float, bool, GtType]:
    """
    Compute mesh volume in cubic meters.

    Returns (volume_m3, watertight, gt_type).
    Raises ValueError if the mesh is not watertight and repair=False.
    """
    working = clean_mesh(mesh)
    watertight = bool(working.is_watertight)

    if watertight:
        volume = abs(float(working.volume))
        return volume, True, "mesh_watertight"

    if not repair:
        raise ValueError(
            "Mesh is not watertight; refusing to treat volume as ground truth. "
            "Pass repair=True to attempt repair, or use convex_hull / voxel pseudo-GT."
        )

    try:
        trimesh.repair.fill_holes(working)
        working.fix_normals()
        working.update_faces(working.nondegenerate_faces())
        working.merge_vertices()
    except Exception as exc:
        raise ValueError(f"Mesh repair failed: {exc}") from exc

    if not working.is_watertight:
        raise ValueError(
            "Mesh could not be repaired to watertight; volume is not reliable ground truth."
        )

    volume = abs(float(working.volume))
    return volume, True, "mesh_repaired"


def compute_convex_hull_volume_m3(mesh: trimesh.Trimesh) -> float:
    """Volume of the mesh convex hull in cubic meters."""
    hull = mesh.convex_hull
    return abs(float(hull.volume))


def compute_voxelized_mesh_volume_m3(mesh: trimesh.Trimesh, voxel_size: float) -> float:
    """Approximate mesh volume by voxelization (meters)."""
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    voxels = mesh.voxelized(pitch=voxel_size)
    return float(voxels.filled_count * (voxel_size ** 3))


def write_gt_volume_json(
    path: str | Path,
    volume_m3: float,
    method: GtType,
    watertight: bool,
    source_mesh: str | Path,
) -> None:
    """Write ground-truth volume metadata JSON."""
    if volume_m3 <= 0:
        raise ValueError(f"volume_m3 must be positive, got {volume_m3}")
    if method not in ("mesh_watertight", "mesh_repaired", "full_reconstruction_pseudo_gt"):
        raise ValueError(f"Invalid gt_type: {method}")

    payload = {
        "volume_m3": float(volume_m3),
        "volume_cm3": float(volume_m3 * 1e6),
        "gt_type": method,
        "watertight": bool(watertight),
        "source_mesh": str(Path(source_mesh).expanduser().resolve()),
    }
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

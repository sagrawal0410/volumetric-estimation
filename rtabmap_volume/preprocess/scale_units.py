"""Unit scaling and scale inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh


UNIT_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
}


@dataclass
class ScaleResult:
    scale_factor: float
    warnings: list[str]
    bbox_diagonal_m: float


def unit_scale_to_meters(unit: str) -> float:
    key = unit.strip().lower()
    if key not in UNIT_TO_METERS:
        raise ValueError(f"Unknown unit: {unit!r}. Supported: {sorted(UNIT_TO_METERS)}")
    return UNIT_TO_METERS[key]


def bbox_diagonal_from_points(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def infer_scale_warnings(bbox_diagonal_m: float) -> list[str]:
    warnings: list[str] = []
    if bbox_diagonal_m > 1000:
        warnings.append(
            f"Bounding box diagonal is {bbox_diagonal_m:.1f} m — likely input is in millimeters, not meters."
        )
    elif bbox_diagonal_m > 100:
        warnings.append(
            f"Bounding box diagonal is {bbox_diagonal_m:.1f} m — verify units; may be centimeters."
        )
    if 0 < bbox_diagonal_m < 0.001:
        warnings.append(
            f"Bounding box diagonal is {bbox_diagonal_m:.6f} m — suspiciously small; check units."
        )
    return warnings


def load_known_scale_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def compute_known_scale_factor(data: dict[str, Any]) -> float:
    """Compute scale factor from known distance between two points."""
    known_m = float(data["known_distance_m"])
    a = np.asarray(data["point_a"], dtype=np.float64)
    b = np.asarray(data["point_b"], dtype=np.float64)
    measured = float(np.linalg.norm(b - a))
    if measured <= 0:
        raise ValueError("Measured distance between scale points is zero")
    return known_m / measured


def scale_mesh_to_meters(mesh: trimesh.Trimesh, unit: str, extra_scale: float = 1.0) -> trimesh.Trimesh:
    factor = unit_scale_to_meters(unit) * extra_scale
    if factor == 1.0:
        return mesh
    scaled = mesh.copy()
    scaled.apply_scale(factor)
    return scaled


def scale_point_cloud_to_meters(
    pcd: o3d.geometry.PointCloud, unit: str, extra_scale: float = 1.0
) -> o3d.geometry.PointCloud:
    factor = unit_scale_to_meters(unit) * extra_scale
    if factor == 1.0:
        return pcd
    scaled = o3d.geometry.PointCloud(pcd)
    pts = np.asarray(scaled.points)
    scaled.points = o3d.utility.Vector3dVector(pts * factor)
    return scaled


def apply_scaling(
    mesh: trimesh.Trimesh | None,
    pcd: o3d.geometry.PointCloud | None,
    unit: str,
    known_scale_json: str | Path | None = None,
) -> ScaleResult:
    extra = 1.0
    warnings: list[str] = []

    if known_scale_json:
        data = load_known_scale_json(known_scale_json)
        extra = compute_known_scale_factor(data)
        warnings.append(f"Applied known-scale correction factor: {extra:.6f}")

    if mesh is not None:
        mesh_scaled = scale_mesh_to_meters(mesh, unit, extra)
        diag = bbox_diagonal_from_points(np.asarray(mesh_scaled.vertices))
        warnings.extend(infer_scale_warnings(diag))
        # Mutate in place for caller convenience
        mesh.vertices[:] = mesh_scaled.vertices
        return ScaleResult(unit_scale_to_meters(unit) * extra, warnings, diag)

    if pcd is not None:
        pcd_scaled = scale_point_cloud_to_meters(pcd, unit, extra)
        pts = np.asarray(pcd_scaled.points)
        pcd.points = o3d.utility.Vector3dVector(pts)
        diag = bbox_diagonal_from_points(pts)
        warnings.extend(infer_scale_warnings(diag))
        return ScaleResult(unit_scale_to_meters(unit) * extra, warnings, diag)

    return ScaleResult(1.0, warnings, 0.0)

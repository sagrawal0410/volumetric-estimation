"""ROI cropping: AABB and OBB."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh

from rtabmap_volume.preprocess.scale_units import unit_scale_to_meters


@dataclass
class ROIBox:
    kind: str  # aabb or obb
    data: dict[str, Any]


def load_roi_json(path: str | Path) -> ROIBox:
    with Path(path).open() as f:
        data = json.load(f)
    if "min" in data and "max" in data:
        return ROIBox("aabb", data)
    if "center" in data and "R" in data and "extent" in data:
        return ROIBox("obb", data)
    raise ValueError("ROI JSON must contain AABB (min/max) or OBB (center/R/extent) fields")


def _scale_roi(data: dict[str, Any]) -> dict[str, Any]:
    unit = data.get("units", "m")
    factor = unit_scale_to_meters(unit)
    scaled = dict(data)
    if "min" in data:
        scaled["min"] = (np.asarray(data["min"]) * factor).tolist()
        scaled["max"] = (np.asarray(data["max"]) * factor).tolist()
    if "center" in data:
        scaled["center"] = (np.asarray(data["center"]) * factor).tolist()
        scaled["extent"] = (np.asarray(data["extent"]) * factor).tolist()
    return scaled


def crop_point_cloud_aabb(pcd: o3d.geometry.PointCloud, mn: np.ndarray, mx: np.ndarray) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    mask = np.all(pts >= mn, axis=1) & np.all(pts <= mx, axis=1)
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        cropped.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        cropped.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
    return cropped


def crop_mesh_aabb(mesh: trimesh.Trimesh, mn: np.ndarray, mx: np.ndarray) -> trimesh.Trimesh:
    bounds = np.array([mn, mx])
    return mesh.slice_plane(bounds[0], [1, 0, 0], cap=True)  # fallback: bbox filter
    # Simpler bbox filter:


def crop_mesh_aabb_simple(mesh: trimesh.Trimesh, mn: np.ndarray, mx: np.ndarray) -> trimesh.Trimesh:
    verts = np.asarray(mesh.vertices)
    face_mask = np.any((verts[mesh.faces] >= mn) & (verts[mesh.faces] <= mx), axis=(1, 2))
    if not face_mask.any():
        return trimesh.Trimesh()
    sub = mesh.submesh([face_mask], append=True)
    if isinstance(sub, list):
        sub = sub[0] if sub else trimesh.Trimesh()
    return sub


def crop_point_cloud_obb(
    pcd: o3d.geometry.PointCloud,
    center: np.ndarray,
    R: np.ndarray,
    extent: np.ndarray,
) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    local = (pts - center) @ R
    half = extent / 2.0
    mask = np.all(np.abs(local) <= half, axis=1)
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        cropped.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        cropped.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
    return cropped


def apply_roi(
    mesh: trimesh.Trimesh | None,
    pcd: o3d.geometry.PointCloud | None,
    roi: ROIBox,
) -> tuple[trimesh.Trimesh | None, o3d.geometry.PointCloud | None]:
    data = _scale_roi(roi.data)
    if roi.kind == "aabb":
        mn = np.asarray(data["min"], dtype=np.float64)
        mx = np.asarray(data["max"], dtype=np.float64)
        new_mesh = crop_mesh_aabb_simple(mesh, mn, mx) if mesh is not None else None
        new_pcd = crop_point_cloud_aabb(pcd, mn, mx) if pcd is not None else None
        return new_mesh, new_pcd

    center = np.asarray(data["center"], dtype=np.float64)
    R = np.asarray(data["R"], dtype=np.float64)
    extent = np.asarray(data["extent"], dtype=np.float64)
    new_pcd = crop_point_cloud_obb(pcd, center, R, extent) if pcd is not None else None
    # OBB mesh crop: transform to local, AABB crop, transform back (approximate via pcd)
    new_mesh = mesh
    if mesh is not None and new_pcd is not None and len(new_pcd.points) > 0:
        # Reconstruct rough mesh crop from filtered vertices
        vert_set = set(map(tuple, np.round(np.asarray(new_pcd.points), 6)))
        vmask = np.array([tuple(np.round(v, 6)) in vert_set for v in mesh.vertices])
        new_mesh = crop_mesh_aabb_simple(mesh, center - extent / 2, center + extent / 2)
    return new_mesh, new_pcd

"""Load meshes and point clouds from various formats."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh


class GeometryType(str, Enum):
    MESH = "mesh"
    POINT_CLOUD = "point_cloud"
    UNKNOWN = "unknown"


@dataclass
class LoadedGeometry:
    geometry_type: GeometryType
    mesh: trimesh.Trimesh | None = None
    point_cloud: o3d.geometry.PointCloud | None = None
    source_path: Path | None = None
    load_warnings: list[str] | None = None

    def __post_init__(self) -> None:
        if self.load_warnings is None:
            self.load_warnings = []


MESH_EXTENSIONS = {".ply", ".obj", ".stl", ".off"}
CLOUD_EXTENSIONS = {".ply", ".pcd", ".xyz", ".las", ".pts"}


def _is_likely_mesh(path: Path, loaded: Any) -> bool:
    if isinstance(loaded, trimesh.Trimesh):
        return len(loaded.faces) > 0
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        return any(len(g.faces) > 0 for g in geoms)
    return False


def _scene_to_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if not meshes:
        raise ValueError("Scene contains no mesh geometry")
    return trimesh.util.concatenate(meshes)


def load_mesh_trimesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        return _scene_to_mesh(loaded)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unexpected trimesh load result: {type(loaded)}")
    return loaded


def load_mesh_open3d(path: Path) -> trimesh.Trimesh:
    mesh_o3d = o3d.io.read_triangle_mesh(str(path))
    if mesh_o3d.is_empty():
        raise ValueError("Open3D loaded empty mesh")
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def load_point_cloud_open3d(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        # PLY might be mesh-only; try reading as mesh and sampling
        mesh_o3d = o3d.io.read_triangle_mesh(str(path))
        if not mesh_o3d.is_empty() and len(mesh_o3d.triangles) > 0:
            pcd = mesh_o3d.sample_points_uniformly(number_of_points=max(10000, len(mesh_o3d.vertices) * 2))
        else:
            raise ValueError(f"Could not load point cloud from {path}")
    return pcd


def trimesh_to_open3d(mesh: trimesh.Trimesh) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    return pcd


def mesh_to_dense_point_cloud(mesh: trimesh.Trimesh, n_points: int = 50000) -> o3d.geometry.PointCloud:
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    n = max(n_points, len(mesh.vertices) * 2)
    return mesh_o3d.sample_points_uniformly(number_of_points=n)


def open3d_to_trimesh(mesh_o3d: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    vertices = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def load_geometry(path: str | Path, prefer_mesh: bool = True) -> LoadedGeometry:
    """Load geometry from file. Tries trimesh first for meshes, Open3D for clouds."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    warnings: list[str] = []
    ext = path.suffix.lower()

    if prefer_mesh and ext in MESH_EXTENSIONS:
        try:
            mesh = load_mesh_trimesh(path)
            if len(mesh.faces) > 0:
                return LoadedGeometry(
                    geometry_type=GeometryType.MESH,
                    mesh=mesh,
                    source_path=path,
                    load_warnings=warnings,
                )
        except Exception as e:
            warnings.append(f"trimesh load failed: {e}")

        try:
            mesh = load_mesh_open3d(path)
            if len(mesh.faces) > 0:
                warnings.append("Loaded mesh via Open3D fallback")
                return LoadedGeometry(
                    geometry_type=GeometryType.MESH,
                    mesh=mesh,
                    source_path=path,
                    load_warnings=warnings,
                )
        except Exception as e:
            warnings.append(f"Open3D mesh load failed: {e}")

    # Try point cloud
    try:
        pcd = load_point_cloud_open3d(path)
        return LoadedGeometry(
            geometry_type=GeometryType.POINT_CLOUD,
            point_cloud=pcd,
            source_path=path,
            load_warnings=warnings,
        )
    except Exception as e:
        warnings.append(f"Point cloud load failed: {e}")

    raise ValueError(f"Could not load geometry from {path}. Warnings: {warnings}")


def inspect_geometry_stats(geom: LoadedGeometry) -> dict[str, Any]:
    """Return basic statistics about loaded geometry."""
    stats: dict[str, Any] = {"geometry_type": geom.geometry_type.value, "warnings": geom.load_warnings or []}

    if geom.geometry_type == GeometryType.MESH and geom.mesh is not None:
        mesh = geom.mesh
        bounds = mesh.bounds
        dims = bounds[1] - bounds[0]
        stats.update(
            {
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "bbox_min": bounds[0].tolist(),
                "bbox_max": bounds[1].tolist(),
                "bbox_dims": dims.tolist(),
                "bbox_volume": float(np.prod(dims)),
                "bbox_diagonal": float(np.linalg.norm(dims)),
                "is_watertight": bool(mesh.is_watertight),
                "is_volume": bool(mesh.is_volume),
            }
        )
    elif geom.point_cloud is not None:
        pts = np.asarray(geom.point_cloud.points)
        if len(pts) == 0:
            stats["point_count"] = 0
        else:
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            dims = mx - mn
            stats.update(
                {
                    "point_count": len(pts),
                    "bbox_min": mn.tolist(),
                    "bbox_max": mx.tolist(),
                    "bbox_dims": dims.tolist(),
                    "bbox_volume": float(np.prod(dims)),
                    "bbox_diagonal": float(np.linalg.norm(dims)),
                }
            )
    return stats

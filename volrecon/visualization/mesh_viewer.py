"""Open3D mesh viewer helper."""

from __future__ import annotations

from pathlib import Path

import open3d as o3d


def show_meshes(
    pred_mesh: Path | None = None,
    gt_mesh: Path | None = None,
    pointcloud: Path | None = None,
) -> None:
    geoms = []
    if pred_mesh and Path(pred_mesh).exists():
        m = o3d.io.read_triangle_mesh(str(pred_mesh))
        m.paint_uniform_color([0.2, 0.6, 1.0])
        m.compute_vertex_normals()
        geoms.append(m)
    if gt_mesh and Path(gt_mesh).exists():
        m = o3d.io.read_triangle_mesh(str(gt_mesh))
        m.paint_uniform_color([1.0, 0.4, 0.2])
        m.compute_vertex_normals()
        geoms.append(m)
    if pointcloud and Path(pointcloud).exists():
        p = o3d.io.read_point_cloud(str(pointcloud))
        geoms.append(p)
    if not geoms:
        raise FileNotFoundError("No geometry to visualize")
    o3d.visualization.draw_geometries(geoms)

"""Offscreen screenshot rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from rtabmap_volume.viz.open3d_viewer import try_offscreen


def _trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    m.compute_vertex_normals()
    return m


def render_geometry_screenshot(
    mesh: trimesh.Trimesh | None = None,
    pcd: o3d.geometry.PointCloud | None = None,
    out_path: Path | None = None,
    width: int = 800,
    height: int = 600,
) -> bool:
    if not try_offscreen():
        return False
    geoms = []
    if pcd is not None and not pcd.is_empty():
        geoms.append(pcd)
    if mesh is not None and len(mesh.faces) > 0:
        geoms.append(_trimesh_to_o3d(mesh))
    if not geoms:
        return False

    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        for g in geoms:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            vis.capture_screen_image(str(out_path))
        vis.destroy_window()
        return True
    except Exception:
        return False

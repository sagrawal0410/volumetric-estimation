"""TSDF fusion on sparse WildRGB-D views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from wildrgbd_volume_benchmark.geometry import invert_T, make_o3d_intrinsic
from wildrgbd_volume_benchmark.scan_io import (
    load_prepared_scene,
    pseudo_gt_comparison_fields,
    resolve_output_dir,
    write_report,
)


def _clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    m.fix_normals()
    return m


def estimate_tsdf_volume(
    prepared_scene_dir: str | Path,
    voxel_length: float = 0.003,
    sdf_trunc: float = 0.015,
    depth_trunc: float = 5.0,
    output_dir: str | Path | None = None,
    repair_mesh: bool = False,
) -> dict[str, Any]:
    import os

    if os.environ.get("WILDRGBD_SKIP_OPEN3D", "0") == "1":
        raise RuntimeError("Open3D disabled (WILDRGBD_SKIP_OPEN3D=1)")
    import open3d as o3d

    scene = load_prepared_scene(prepared_scene_dir)
    out = resolve_output_dir(scene.scene_dir, "tsdf", Path(output_dir) if output_dir else None)
    out.mkdir(parents=True, exist_ok=True)

    h, w = scene.frames[0].depth_m.shape
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for frame in scene.frames:
        depth = frame.depth_m.copy()
        depth[~frame.mask] = 0.0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8)),
            o3d.geometry.Image(depth.astype(np.float32)),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = make_o3d_intrinsic(frame.K, w, h)
        T_object_to_cam = invert_T(frame.T_cam_to_object)
        volume.integrate(rgbd, intrinsic, T_object_to_cam)

    raw_o3d = volume.extract_triangle_mesh()
    raw_mesh = trimesh.Trimesh(
        vertices=np.asarray(raw_o3d.vertices),
        faces=np.asarray(raw_o3d.triangles),
        process=False,
    )
    parts = raw_mesh.split(only_watertight=False)
    cleaned = _clean_mesh(max(parts, key=lambda m: len(m.faces)) if parts else raw_mesh)

    watertight = bool(cleaned.is_watertight)
    volume_m3 = abs(float(cleaned.volume)) if watertight else None
    if not watertight and repair_mesh:
        try:
            trimesh.repair.fill_holes(cleaned)
            cleaned.fix_normals()
            watertight = bool(cleaned.is_watertight)
            if watertight:
                volume_m3 = abs(float(cleaned.volume))
        except Exception:
            pass

    raw_mesh.export(out / "tsdf_mesh_raw.ply")
    cleaned.export(out / "tsdf_mesh_cleaned.ply")

    report: dict[str, Any] = {
        "method": "tsdf",
        "prepared_scene_dir": str(scene.scene_dir),
        "volume_m3": volume_m3,
        "volume_cm3": volume_m3 * 1e6 if volume_m3 is not None else None,
        "watertight": watertight,
        "voxel_length": voxel_length,
        "sdf_trunc": sdf_trunc,
        "status": "ok" if volume_m3 is not None else "invalid_mesh",
        "notes": "Sparse-view TSDF vs full-video pseudo-GT",
    }
    report.update(pseudo_gt_comparison_fields(scene, volume_m3))
    write_report(out / "report.json", report)
    return report

"""TSDF fusion volume estimation using Open3D."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import trimesh

from volume_benchmark.common.geometry import invert_T
from volume_benchmark.common.io import Frame
from volume_benchmark.common.mesh_volume import clean_mesh
from volume_benchmark.methods._io import (
    gt_comparison_fields,
    load_scan_or_raise,
    resolve_output_dir,
    write_report,
)


def _require_open3d():
    if os.environ.get("VOLUME_BENCHMARK_SKIP_OPEN3D", "0") == "1":
        raise RuntimeError(
            "Open3D is disabled (VOLUME_BENCHMARK_SKIP_OPEN3D=1). "
            "Unset it to run TSDF volume estimation."
        )
    import open3d as o3d

    return o3d


def _frame_to_o3d_rgbd(frame: Frame, depth_trunc: float, o3d) -> "o3d.geometry.RGBDImage":
    depth = frame.depth_m.copy()
    depth[~frame.mask] = 0.0
    depth_img = o3d.geometry.Image(depth.astype(np.float32))
    color = o3d.geometry.Image(
        np.stack([frame.mask.astype(np.uint8) * 255] * 3, axis=-1)
    )
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth_img,
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )


def _largest_connected_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if not parts:
        return mesh
    return max(parts, key=lambda m: len(m.faces))


def _o3d_to_trimesh(mesh_o3d) -> trimesh.Trimesh:
    verts = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def estimate_volume_tsdf(
    frames: Sequence[Frame],
    K: np.ndarray,
    voxel_length: float = 0.003,
    sdf_trunc: float = 0.015,
    depth_trunc: float = 5.0,
    repair_mesh: bool = False,
) -> tuple[Optional[float], trimesh.Trimesh, trimesh.Trimesh, bool, bool]:
    """
    Integrate depth frames into TSDF and return mesh volume if watertight.

    Returns (volume_m3 or None, raw_mesh, cleaned_mesh, watertight, repaired_used).
    """
    o3d = _require_open3d()
    if not frames:
        raise ValueError("At least one frame is required")

    height, width = frames[0].depth_m.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=int(width),
        height=int(height),
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for frame in frames:
        rgbd = _frame_to_o3d_rgbd(frame, depth_trunc=depth_trunc, o3d=o3d)
        T_object_to_cam = invert_T(frame.T_cam_to_object)
        volume.integrate(rgbd, intrinsic, T_object_to_cam)

    raw_o3d = volume.extract_triangle_mesh()
    raw_o3d.compute_vertex_normals()
    if len(raw_o3d.triangles) == 0:
        raise ValueError("TSDF extraction produced an empty mesh")

    raw_mesh = _o3d_to_trimesh(raw_o3d)
    cleaned = clean_mesh(_largest_connected_component(raw_mesh))

    watertight = bool(cleaned.is_watertight)
    repaired_used = False
    vol: Optional[float] = None

    if watertight:
        vol = abs(float(cleaned.volume))
    elif repair_mesh:
        try:
            trimesh.repair.fill_holes(cleaned)
            cleaned.fix_normals()
            cleaned.merge_vertices()
            repaired_used = True
            watertight = bool(cleaned.is_watertight)
            if watertight:
                vol = abs(float(cleaned.volume))
        except Exception:
            pass

    return vol, raw_mesh, cleaned, watertight, repaired_used


def estimate_tsdf_volume(
    scan_dir: str | Path,
    output_dir: str | Path | None = None,
    voxel_length: float = 0.003,
    sdf_trunc: float = 0.015,
    depth_trunc: float = 5.0,
    repair_mesh: bool = False,
) -> dict[str, Any]:
    """Run TSDF fusion on a prepared scan and write mesh outputs + report."""
    scan = load_scan_or_raise(scan_dir)
    out = resolve_output_dir(scan.scan_dir, "tsdf", output_dir)
    out.mkdir(parents=True, exist_ok=True)

    volume_m3, raw_mesh, cleaned_mesh, watertight, repaired_used = estimate_volume_tsdf(
        scan.frames,
        scan.K,
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc,
        repair_mesh=repair_mesh,
    )

    raw_mesh.export(out / "tsdf_mesh_raw.ply")
    cleaned_mesh.export(out / "tsdf_mesh_cleaned.ply")

    report: dict[str, Any] = {
        "method": "tsdf",
        "scan_dir": str(scan.scan_dir),
        "volume_m3": volume_m3,
        "volume_cm3": volume_m3 * 1e6 if volume_m3 is not None else None,
        "watertight": watertight,
        "repaired_used": repaired_used,
        "voxel_length": voxel_length,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
        "outputs": {
            "tsdf_mesh_raw": str(out / "tsdf_mesh_raw.ply"),
            "tsdf_mesh_cleaned": str(out / "tsdf_mesh_cleaned.ply"),
        },
    }
    if not watertight and volume_m3 is None:
        report["warning"] = (
            "Extracted mesh is not watertight; volume_m3 is None. "
            "Pass repair_mesh=True to attempt repair."
        )
    report.update(gt_comparison_fields(scan, volume_m3))
    write_report(out / "report.json", report)
    return report

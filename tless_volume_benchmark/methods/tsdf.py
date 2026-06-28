"""TSDF fusion volume estimation using Open3D."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from tless_volume_benchmark.geometry import create_o3d_intrinsic_from_K, invert_T
from tless_volume_benchmark.mesh_volume import clean_mesh
from tless_volume_benchmark.scan_io import (
    gt_comparison_fields,
    load_prepared_scan,
    resolve_output_dir,
    write_report,
)


def _frame_rgbd(frame, depth_trunc: float, o3d):
    depth = frame.depth_m.copy()
    depth[~frame.mask] = 0.0
    depth_img = o3d.geometry.Image(depth.astype(np.float32))
    color = o3d.geometry.Image(
        np.stack([frame.mask.astype(np.uint8) * 255] * 3, axis=-1)
    )
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth_img, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
    )


def _largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    return max(parts, key=lambda m: len(m.faces)) if parts else mesh


def estimate_tsdf(
    scan_dir: str | Path,
    voxel_length: float = 0.002,
    sdf_trunc: float = 0.010,
    depth_trunc: float = 5.0,
    output_dir: str | Path | None = None,
    repair_mesh: bool = False,
) -> dict[str, Any]:
    import os

    if os.environ.get("TLESS_SKIP_OPEN3D", "0") == "1":
        raise RuntimeError(
            "Open3D disabled (TLESS_SKIP_OPEN3D=1). Run without tsdf or fix Open3D install."
        )
    try:
        import open3d as o3d
    except Exception as exc:
        raise ImportError(
            "Open3D failed to import (common on Apple Silicon with wrong-arch wheels). "
            "Recreate the venv with native arm64 Python: "
            "  rm -rf .venv && python3 -m venv .venv && pip install -r requirements.txt\n"
            "Or skip TSDF: --methods convex_hull voxel_carving"
        ) from exc

    scan = load_prepared_scan(scan_dir)
    out = resolve_output_dir(scan.scan_dir, "tsdf", Path(output_dir) if output_dir else None)
    out.mkdir(parents=True, exist_ok=True)

    h, w = scan.frames[0].depth_m.shape
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for frame in scan.frames:
        intrinsic = create_o3d_intrinsic_from_K(frame.K, w, h)
        rgbd = _frame_rgbd(frame, depth_trunc, o3d)
        T_object_to_cam = invert_T(frame.T_cam_to_object)
        volume.integrate(rgbd, intrinsic, T_object_to_cam)

    raw_o3d = volume.extract_triangle_mesh()
    raw_o3d.compute_vertex_normals()
    if len(raw_o3d.triangles) == 0:
        raise ValueError("TSDF extraction produced an empty mesh")

    raw_mesh = trimesh.Trimesh(
        vertices=np.asarray(raw_o3d.vertices),
        faces=np.asarray(raw_o3d.triangles),
        process=False,
    )
    cleaned = clean_mesh(_largest_component(raw_mesh))

    watertight = bool(cleaned.is_watertight)
    repaired_used = False
    volume_m3 = abs(float(cleaned.volume)) if watertight else None

    if not watertight and repair_mesh:
        try:
            trimesh.repair.fill_holes(cleaned)
            cleaned.fix_normals()
            cleaned.merge_vertices()
            repaired_used = True
            watertight = bool(cleaned.is_watertight)
            if watertight:
                volume_m3 = abs(float(cleaned.volume))
        except Exception:
            pass

    raw_mesh.export(out / "tsdf_mesh_raw.ply")
    cleaned.export(out / "tsdf_mesh_cleaned.ply")

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
        "status": "ok" if volume_m3 is not None else "invalid_mesh",
        "outputs": {
            "tsdf_mesh_raw": str(out / "tsdf_mesh_raw.ply"),
            "tsdf_mesh_cleaned": str(out / "tsdf_mesh_cleaned.ply"),
        },
    }
    if volume_m3 is None:
        report["warning"] = "Mesh not watertight; volume unavailable unless repair_mesh=True"
    report.update(gt_comparison_fields(scan, volume_m3))
    write_report(out / "report.json", report)
    return report

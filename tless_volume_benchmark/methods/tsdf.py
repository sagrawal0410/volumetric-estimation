"""TSDF fusion volume estimation using Open3D."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from tless_volume_benchmark.geometry import create_o3d_intrinsic_from_K, invert_T
from tless_volume_benchmark.mesh_volume import clean_mesh
from tless_volume_benchmark.scan_io import (
    PreparedScan,
    gt_comparison_fields,
    load_prepared_scan,
    resolve_output_dir,
    write_report,
)


def _require_open3d():
    if os.environ.get("TLESS_SKIP_OPEN3D", "0") == "1":
        raise RuntimeError(
            "Open3D disabled (TLESS_SKIP_OPEN3D=1). Run without tsdf or fix Open3D install."
        )
    try:
        import open3d as o3d
    except Exception as exc:
        raise ImportError(
            "Open3D failed to import. Try: pip install -U 'open3d>=0.17,<0.20'\n"
            "Or skip TSDF: --methods convex_hull voxel_carving"
        ) from exc
    return o3d


def _sanitize_depth(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    depth = np.ascontiguousarray(depth_m, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0.0
    depth[~mask.astype(bool)] = 0.0
    return depth


def _sanitize_extrinsic(T: np.ndarray) -> np.ndarray:
    extr = np.ascontiguousarray(T, dtype=np.float64)
    if extr.shape != (4, 4):
        raise ValueError(f"Extrinsic must be (4, 4), got {extr.shape}")
    if not np.all(np.isfinite(extr)):
        raise ValueError("Extrinsic contains non-finite values")
    return extr


def _frame_rgbd(frame, depth_trunc: float, o3d, height: int, width: int):
    depth = _sanitize_depth(frame.depth_m, frame.mask)
    if depth.shape[0] != height or depth.shape[1] != width:
        height, width = depth.shape
    # Plain black color (mask-as-color can upset some Open3D builds).
    color = o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8))
    depth_img = o3d.geometry.Image(depth)
    return (
        o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth_img,
            depth_scale=1.0,
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        ),
        height,
        width,
    )


def _object_bounds_from_scan(scan: PreparedScan, padding: float) -> tuple[np.ndarray, np.ndarray]:
    from tless_volume_benchmark.methods.convex_hull import fuse_points

    points = fuse_points(scan)
    if points.shape[0] == 0:
        raise ValueError("No valid depth points for TSDF volume bounds")
    lo = points.min(axis=0) - padding
    hi = points.max(axis=0) + padding
    return lo, hi


def _make_tsdf_volume(
    o3d,
    scan: PreparedScan,
    voxel_length: float,
    sdf_trunc: float,
) -> tuple[Any, dict[str, Any]]:
    color_type = o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    backend = os.environ.get("TLESS_TSDF_BACKEND", "uniform").lower()

    if sdf_trunc < voxel_length:
        raise ValueError(
            f"sdf_trunc ({sdf_trunc}) should be >= voxel_length ({voxel_length}) for stable TSDF"
        )

    if backend == "scalable":
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_length),
            sdf_trunc=float(sdf_trunc),
            color_type=color_type,
        )
        return volume, {"tsdf_backend": "scalable"}

    padding = max(3.0 * voxel_length, 0.02)
    lo, hi = _object_bounds_from_scan(scan, padding)
    extent = hi - lo
    length = float(np.max(extent))
    if length <= 0 or not np.isfinite(length):
        raise ValueError(f"Invalid TSDF volume extent: {extent}")

    resolution = int(np.ceil(length / float(voxel_length)))
    resolution = max(32, min(resolution, int(os.environ.get("TLESS_TSDF_MAX_RESOLUTION", "384"))))

    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length,
        resolution,
        float(sdf_trunc),
        color_type,
        origin=lo.astype(np.float64).tolist(),
    )
    meta = {
        "tsdf_backend": "uniform",
        "volume_length_m": length,
        "volume_resolution": resolution,
        "volume_origin_m": lo.tolist(),
    }
    return volume, meta


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
    verbose: bool = True,
) -> dict[str, Any]:
    o3d = _require_open3d()
    scan = load_prepared_scan(scan_dir)
    out = resolve_output_dir(scan.scan_dir, "tsdf", Path(output_dir) if output_dir else None)
    out.mkdir(parents=True, exist_ok=True)

    volume, vol_meta = _make_tsdf_volume(o3d, scan, voxel_length, sdf_trunc)
    if verbose:
        print(f"  TSDF backend: {vol_meta.get('tsdf_backend', 'unknown')}", flush=True)
        if vol_meta.get("tsdf_backend") == "uniform":
            print(
                f"  uniform cube length={vol_meta['volume_length_m']:.3f} m, "
                f"resolution={vol_meta['volume_resolution']}",
                flush=True,
            )

    for idx, frame in enumerate(scan.frames):
        if verbose:
            print(f"  integrating frame {idx}...", flush=True)
        h, w = frame.depth_m.shape
        rgbd, h, w = _frame_rgbd(frame, depth_trunc, o3d, h, w)
        intrinsic = create_o3d_intrinsic_from_K(frame.K, w, h)
        extrinsic = _sanitize_extrinsic(invert_T(frame.T_cam_to_object))
        volume.integrate(rgbd, intrinsic, extrinsic)

    if verbose:
        print("  extracting mesh...", flush=True)
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
        **vol_meta,
    }
    if volume_m3 is None:
        report["warning"] = "Mesh not watertight; volume unavailable unless repair_mesh=True"
    report.update(gt_comparison_fields(scan, volume_m3))
    write_report(out / "report.json", report)
    return report

"""Open3D TSDF fusion (optional backend — set TLESS_TSDF_BACKEND=open3d)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import trimesh

from tless_volume_benchmark.geometry import create_o3d_intrinsic_from_K, invert_T
from tless_volume_benchmark.mesh_volume import clean_mesh
from tless_volume_benchmark.scan_io import PreparedScan


def _require_open3d():
    if os.environ.get("TLESS_SKIP_OPEN3D", "0") == "1":
        raise RuntimeError("Open3D disabled (TLESS_SKIP_OPEN3D=1)")
    import open3d as o3d

    return o3d


def _sanitize_depth(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    depth = np.ascontiguousarray(depth_m, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0.0
    depth[~mask.astype(bool)] = 0.0
    return depth


def _object_bounds_from_scan(scan: PreparedScan, padding: float) -> tuple[np.ndarray, np.ndarray]:
    from tless_volume_benchmark.methods.convex_hull import fuse_points

    points = fuse_points(scan)
    if points.shape[0] == 0:
        raise ValueError("No valid depth points for TSDF volume bounds")
    lo = points.min(axis=0) - padding
    hi = points.max(axis=0) + padding
    return lo, hi


def run_open3d_tsdf(
    scan: PreparedScan,
    voxel_length: float,
    sdf_trunc: float,
    depth_trunc: float,
    repair_mesh: bool = False,
    verbose: bool = True,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict[str, Any], float | None]:
    o3d = _require_open3d()
    color_type = o3d.pipelines.integration.TSDFVolumeColorType.NoColor

    padding = max(3.0 * voxel_length, 0.02)
    lo, hi = _object_bounds_from_scan(scan, padding)
    length = float(np.max(hi - lo))
    resolution = int(np.ceil(length / float(voxel_length)))
    resolution = max(32, min(resolution, int(os.environ.get("TLESS_TSDF_MAX_RESOLUTION", "128"))))

    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length,
        resolution,
        float(sdf_trunc),
        color_type,
        origin=lo.astype(np.float64).tolist(),
    )
    meta: dict[str, Any] = {
        "tsdf_backend": "open3d_uniform",
        "volume_length_m": length,
        "volume_resolution": resolution,
        "volume_origin_m": lo.tolist(),
    }

    for idx, frame in enumerate(scan.frames):
        if verbose:
            print(f"  open3d integrate frame {idx}...", flush=True)
        depth = _sanitize_depth(frame.depth_m, frame.mask)
        h, w = depth.shape
        color = o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8))
        depth_img = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth_img,
            depth_scale=1.0,
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )
        intrinsic = create_o3d_intrinsic_from_K(frame.K, w, h)
        extrinsic = np.ascontiguousarray(invert_T(frame.T_cam_to_object), dtype=np.float64)
        volume.integrate(rgbd, intrinsic, extrinsic)

    raw_o3d = volume.extract_triangle_mesh()
    raw_o3d.compute_vertex_normals()
    if len(raw_o3d.triangles) == 0:
        raise ValueError("Open3D TSDF produced empty mesh")

    raw_mesh = trimesh.Trimesh(
        vertices=np.asarray(raw_o3d.vertices),
        faces=np.asarray(raw_o3d.triangles),
        process=False,
    )
    parts = raw_mesh.split(only_watertight=False)
    cleaned = clean_mesh(max(parts, key=lambda m: len(m.faces)) if parts else raw_mesh)

    watertight = bool(cleaned.is_watertight)
    volume_m3 = abs(float(cleaned.volume)) if watertight else None
    if not watertight and repair_mesh:
        try:
            trimesh.repair.fill_holes(cleaned)
            cleaned.fix_normals()
            cleaned.merge_vertices()
            watertight = bool(cleaned.is_watertight)
            if watertight:
                volume_m3 = abs(float(cleaned.volume))
        except Exception:
            pass

    meta["watertight"] = watertight
    meta["volume_source"] = "mesh_watertight" if watertight else "open3d_non_watertight"
    return raw_mesh, cleaned, meta, volume_m3

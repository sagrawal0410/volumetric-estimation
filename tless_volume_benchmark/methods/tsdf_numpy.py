"""Pure NumPy TSDF fusion (no Open3D). Default backend for tless_volume_benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from tless_volume_benchmark.geometry import invert_T, transform_points
from tless_volume_benchmark.scan_io import PreparedScan


def _sanitize_depth(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    depth = np.ascontiguousarray(depth_m, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0.0
    depth[~mask.astype(bool)] = 0.0
    return depth


def _grid_bounds(scan: PreparedScan, voxel_length: float, padding: float) -> tuple[np.ndarray, np.ndarray]:
    from tless_volume_benchmark.methods.convex_hull import fuse_points

    points = fuse_points(scan)
    if points.shape[0] == 0:
        raise ValueError("No valid depth points for TSDF bounds")
    pad = max(padding, 3.0 * voxel_length)
    lo = points.min(axis=0) - pad
    hi = points.max(axis=0) + pad
    return lo.astype(np.float64), hi.astype(np.float64)


def _sample_depth_nearest(depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    ui = np.clip(np.round(u).astype(np.int32), 0, w - 1)
    vi = np.clip(np.round(v).astype(np.int32), 0, h - 1)
    return depth[vi, ui]


def fuse_tsdf_grid(
    scan: PreparedScan,
    voxel_length: float = 0.002,
    sdf_trunc: float = 0.010,
    depth_trunc: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Fuse masked depth frames into a uniform TSDF grid in object coordinates.

    Returns tsdf (nx,ny,nz), weight grid, origin (3,), metadata.
    """
    lo, hi = _grid_bounds(scan, voxel_length, padding=0.02)
    extent = hi - lo
    dims = np.maximum(1, np.ceil(extent / voxel_length).astype(int))
    max_res = int(__import__("os").environ.get("TLESS_TSDF_MAX_RESOLUTION", "128"))
    if int(np.max(dims)) > max_res:
        scale = float(np.max(dims)) / max_res
        voxel_length = float(voxel_length) * scale
        dims = np.maximum(1, np.ceil(extent / voxel_length).astype(int))

    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    tsdf = np.ones((nx, ny, nz), dtype=np.float32)
    weight = np.zeros((nx, ny, nz), dtype=np.float32)

    xs = lo[0] + (np.arange(nx, dtype=np.float64) + 0.5) * voxel_length
    ys = lo[1] + (np.arange(ny, dtype=np.float64) + 0.5) * voxel_length
    zs = lo[2] + (np.arange(nz, dtype=np.float64) + 0.5) * voxel_length
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    chunk = int(__import__("os").environ.get("TLESS_TSDF_CHUNK", "50000"))
    trunc = float(sdf_trunc)

    for frame in scan.frames:
        depth = _sanitize_depth(frame.depth_m, frame.mask)
        mask = frame.mask.astype(bool)
        h, w = depth.shape
        K = frame.K
        T_oc = invert_T(frame.T_cam_to_object)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        for start in range(0, centers.shape[0], chunk):
            end = min(start + chunk, centers.shape[0])
            pts = centers[start:end]
            cam = transform_points(pts, T_oc)
            x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
            in_front = z > 1e-4
            u = fx * x / np.maximum(z, 1e-9) + cx
            v = fy * y / np.maximum(z, 1e-9) + cy
            in_img = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not np.any(in_img):
                continue

            idx_local = np.where(in_img)[0]
            u_i = u[in_img]
            v_i = v[in_img]
            z_i = z[in_img]
            ui = np.clip(np.round(u_i).astype(np.int32), 0, w - 1)
            vi = np.clip(np.round(v_i).astype(np.int32), 0, h - 1)
            d_i = depth[vi, ui]
            m_i = mask[vi, ui]
            valid = m_i & np.isfinite(d_i) & (d_i > 0.01) & (d_i <= depth_trunc)
            if not np.any(valid):
                continue

            idx_valid = idx_local[valid]
            sdf = np.clip(d_i[valid] - z_i[valid], -trunc, trunc).astype(np.float32)

            w_old = weight.ravel()[idx_valid]
            t_old = tsdf.ravel()[idx_valid]
            w_new = w_old + 1.0
            t_new = (t_old * w_old + sdf) / np.maximum(w_new, 1e-9)
            weight.ravel()[idx_valid] = w_new
            tsdf.ravel()[idx_valid] = t_new

    meta = {
        "tsdf_backend": "numpy",
        "grid_dims": [nx, ny, nz],
        "voxel_length_m": float(voxel_length),
        "origin_m": lo.tolist(),
        "extent_m": extent.tolist(),
    }
    return tsdf, weight, lo, meta


def tsdf_grid_to_mesh(
    tsdf: np.ndarray,
    origin: np.ndarray,
    voxel_length: float,
) -> trimesh.Trimesh:
    """Extract mesh at TSDF zero level set."""
    if not np.any(tsdf < 0):
        raise ValueError("TSDF grid has no inside voxels (tsdf < 0)")

    try:
        from skimage import measure

        verts, faces, _, _ = measure.marching_cubes(
            tsdf.astype(np.float64),
            level=0.0,
            spacing=(float(voxel_length), float(voxel_length), float(voxel_length)),
        )
        verts = verts + origin.reshape(1, 3)
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    except ImportError:
        pass
    except Exception:
        pass

    inside = tsdf < 0
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = transform[1, 1] = transform[2, 2] = float(voxel_length)
    transform[:3, 3] = origin
    vg = trimesh.voxel.VoxelGrid(
        trimesh.voxel.encoding.DenseEncoding(inside.astype(bool)),
        transform=transform,
    )
    return vg.marching_cubes


def tsdf_inside_volume_m3(tsdf: np.ndarray, weight: np.ndarray, voxel_length: float) -> float:
    """Voxel-count volume where fused TSDF is inside surface and observed."""
    inside = (tsdf < 0) & (weight > 0)
    return float(np.count_nonzero(inside)) * (voxel_length ** 3)


def run_numpy_tsdf(
    scan: PreparedScan,
    voxel_length: float,
    sdf_trunc: float,
    depth_trunc: float,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict[str, Any], float | None]:
    """
    Returns raw_mesh, cleaned_mesh, meta, volume_m3 (from watertight mesh or voxel count).
    """
    from tless_volume_benchmark.mesh_volume import clean_mesh

    tsdf, weight, origin, meta = fuse_tsdf_grid(
        scan, voxel_length=voxel_length, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc
    )
    raw_mesh = tsdf_grid_to_mesh(tsdf, origin, voxel_length)
    parts = raw_mesh.split(only_watertight=False)
    cleaned = clean_mesh(max(parts, key=lambda m: len(m.faces)) if parts else raw_mesh)

    watertight = bool(cleaned.is_watertight)
    volume_m3 = abs(float(cleaned.volume)) if watertight and cleaned.volume > 0 else None
    if volume_m3 is None:
        volume_m3 = tsdf_inside_volume_m3(tsdf, weight, voxel_length)
        meta["volume_source"] = "tsdf_voxel_count_fallback"
    else:
        meta["volume_source"] = "mesh_watertight"

    meta["watertight"] = watertight
    return raw_mesh, cleaned, meta, volume_m3

"""Height-field bulk envelope volume above a support plane."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy import ndimage

from rtabmap_volume.config import HeightfieldConfig
from rtabmap_volume.preprocess.plane_removal import segment_plane
from rtabmap_volume.config import PlaneRemovalConfig
from rtabmap_volume.volume.mesh_volume import VolumeEstimate, _liters


def _plane_frame(plane: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a, b, c, d = plane
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    origin = points.mean(axis=0)
    # Project origin onto plane
    t = -(origin @ normal + d) / (normal @ normal)
    origin = origin + t * normal
    arb = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arb)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(normal, u)
    return origin, u, v, normal


def compute_heightfield_volume(
    pcd: o3d.geometry.PointCloud,
    cfg: HeightfieldConfig | None = None,
    plane_model: np.ndarray | None = None,
    plane_cfg: PlaneRemovalConfig | None = None,
) -> VolumeEstimate:
    cfg = cfg or HeightfieldConfig()
    plane_cfg = plane_cfg or PlaneRemovalConfig()
    pts = np.asarray(pcd.points)
    if len(pts) < 10:
        return VolumeEstimate("heightfield_volume", None, None, False, ["Too few points"])

    if plane_model is None:
        plane_result = segment_plane(pcd, plane_cfg)
        plane_model = plane_result.plane_model

    origin, u, v, normal = _plane_frame(plane_model, pts)
    local = pts - origin
    x = local @ u
    y = local @ v
    h = local @ normal

    res = cfg.grid_resolution_m
    x_min, y_min = x.min(), y.min()
    nx = max(int(np.ceil((x.max() - x_min) / res)), 1)
    ny = max(int(np.ceil((y.max() - y_min) / res)), 1)
    cell_area = res * res

    height_grid = np.full((nx, ny), np.nan)
    count_grid = np.zeros((nx, ny), dtype=int)

    ix = np.clip(((x - x_min) / res).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / res).astype(int), 0, ny - 1)

    for i in range(len(h)):
        ci, cj = ix[i], iy[i]
        count_grid[ci, cj] += 1
        if np.isnan(height_grid[ci, cj]):
            height_grid[ci, cj] = h[i]
        else:
            if cfg.height_stat == "max":
                height_grid[ci, cj] = max(height_grid[ci, cj], h[i])
            elif cfg.height_stat in ("p95", "p90"):
                # accumulate for percentile — simplified: running max for sparse
                height_grid[ci, cj] = max(height_grid[ci, cj], h[i])

    valid_mask = count_grid >= cfg.min_points_per_cell
    height_grid[~valid_mask] = np.nan

    if cfg.hole_fill_method == "nearest" and np.any(~np.isnan(height_grid)):
        nan_mask = np.isnan(height_grid)
        if np.any(nan_mask):
            ind = ndimage.distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
            height_grid[nan_mask] = height_grid[tuple(ind[:, nan_mask])]

    base = cfg.base_height_m
    heights = np.maximum(height_grid - base, 0.0)
    heights[np.isnan(heights)] = 0.0
    vol = float(np.nansum(heights) * cell_area)

    return VolumeEstimate(
        "heightfield_volume",
        vol,
        _liters(vol),
        True,
        ["Height-field estimates visible/bulk envelope volume, not hidden material volume"],
        metadata={"grid_resolution_m": res, "height_stat": cfg.height_stat},
    )

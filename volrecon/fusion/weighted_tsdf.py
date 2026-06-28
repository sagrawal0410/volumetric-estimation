"""Uncertainty-weighted TSDF fusion."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from skimage import measure
import yaml

from volrecon.fusion.probabilistic_occupancy import LogOddsOccupancyGrid, OccupancyConfig
from volrecon.fusion.robust_kernels import huber_weight, tukey_weight
from volrecon.geometry.transforms import transform_points
from volrecon.io.json_io import write_json

logger = logging.getLogger(__name__)


@dataclass
class WeightedTSDFConfig:
    voxel_length_m: float = 0.003
    sdf_trunc_m: float = 0.015
    min_depth_m: float = 0.1
    max_depth_m: float = 2.0
    W_max: float = 100.0
    W_min_robust: float = 0.5
    min_weight_for_mesh: float = 2.0
    max_weight_per_obs: float = 5.0
    chunk_size: int = 32
    integrate_color: bool = False
    mesh_cleanup: bool = True
    use_occupancy: bool = True
    occupancy_threshold: float = 0.5
    robust_kernel: str = "huber"
    robust_delta: float = 0.25


class DenseChunkedWeightedTSDF:
    """Dense voxel-grid weighted TSDF integrator with chunked updates."""

    def __init__(self, bounds: np.ndarray, cfg: WeightedTSDFConfig) -> None:
        self.cfg = cfg
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.voxel_size = float(cfg.voxel_length_m)
        self.origin = self.bounds[0].copy()
        extent = self.bounds[1] - self.bounds[0]
        self.dims = np.maximum(np.ceil(extent / self.voxel_size).astype(int), 1)
        d, h, w = self.dims  # z, y, x ordering for marching cubes

        self.tsdf = np.ones((d, h, w), dtype=np.float32)
        self.weight = np.zeros((d, h, w), dtype=np.float32)
        self.variance = np.full((d, h, w), np.inf, dtype=np.float32)
        self.color_sum = np.zeros((d, h, w, 3), dtype=np.float32)
        self.color_weight = np.zeros((d, h, w), dtype=np.float32)
        self.obs_count = np.zeros((d, h, w), dtype=np.uint16)

        self.occupancy = LogOddsOccupancyGrid((d, h, w)) if cfg.use_occupancy else None
        self._frames: list[dict[str, Any]] = []

    def world_to_voxel(self, pts_world: np.ndarray) -> np.ndarray:
        return (pts_world - self.origin) / self.voxel_size

    def voxel_centers_chunk(self, z0: int, y0: int, x0: int, cz: int, cy: int, cx: int) -> np.ndarray:
        zz = np.arange(z0, min(z0 + cz, self.dims[0]))
        yy = np.arange(y0, min(y0 + cy, self.dims[1]))
        xx = np.arange(x0, min(x0 + cx, self.dims[2]))
        Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
        idx = np.stack([Z, Y, X], axis=-1).reshape(-1, 3)
        centers = self.origin + (idx + 0.5) * self.voxel_size
        return idx, centers

    def integrate_view(
        self,
        depth_m: np.ndarray,
        weight_map: np.ndarray,
        K: np.ndarray,
        T_cam_world: np.ndarray,
        rgb: np.ndarray | None = None,
        view_id: str | None = None,
    ) -> None:
        cfg = self.cfg
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        T_cw = np.asarray(T_cam_world, dtype=np.float64).reshape(4, 4)
        h_img, w_img = depth_m.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        cs = cfg.chunk_size

        for z0 in range(0, self.dims[0], cs):
            for y0 in range(0, self.dims[1], cs):
                for x0 in range(0, self.dims[2], cs):
                    idx, centers_w = self.voxel_centers_chunk(z0, y0, x0, cs, cs, cs)
                    pts_c = transform_points(T_cw, centers_w)
                    z_cam = pts_c[:, 2]
                    valid_z = z_cam > cfg.min_depth_m
                    u = fx * pts_c[:, 0] / np.maximum(z_cam, 1e-6) + cx
                    v = fy * pts_c[:, 1] / np.maximum(z_cam, 1e-6) + cy
                    ui = np.round(u).astype(int)
                    vi = np.round(v).astype(int)
                    in_img = (ui >= 0) & (ui < w_img) & (vi >= 0) & (vi < h_img) & valid_z

                    if not np.any(in_img):
                        continue

                    sel = np.where(in_img)[0]
                    depth_obs = depth_m[vi[sel], ui[sel]]
                    w_obs = weight_map[vi[sel], ui[sel]]
                    good = (depth_obs > cfg.min_depth_m) & (depth_obs < cfg.max_depth_m) & (w_obs > 0)
                    if not np.any(good):
                        continue

                    sel = sel[good]
                    depth_obs = depth_obs[good]
                    w_obs = np.clip(w_obs[good], 0.0, cfg.max_weight_per_obs)
                    z_cam_sel = z_cam[sel]
                    sdf = depth_obs - z_cam_sel
                    in_trunc = (sdf >= -cfg.sdf_trunc_m) & (sdf <= cfg.sdf_trunc_m)
                    if not np.any(in_trunc):
                        continue

                    sel = sel[in_trunc]
                    sdf = sdf[in_trunc]
                    w_obs = w_obs[in_trunc]
                    tsdf_obs = np.clip(sdf / cfg.sdf_trunc_m, -1.0, 1.0).astype(np.float32)
                    ui_sel = ui[sel]
                    vi_sel = vi[sel]

                    for j, vidx in enumerate(sel):
                        iz, iy, ix = int(idx[vidx, 0]), int(idx[vidx, 1]), int(idx[vidx, 2])
                        w_old = float(self.weight[iz, iy, ix])
                        t_old = float(self.tsdf[iz, iy, ix])
                        w_pix = float(w_obs[j])
                        if w_old > cfg.W_min_robust:
                            residual = float(tsdf_obs[j] - t_old)
                            if cfg.robust_kernel == "tukey":
                                w_pix *= float(tukey_weight(np.array([residual]), cfg.robust_delta)[0])
                            else:
                                w_pix *= float(huber_weight(np.array([residual]), cfg.robust_delta)[0])
                        w_new = min(w_old + w_pix, cfg.W_max)
                        if w_new <= 0:
                            continue
                        t_new = (w_old * t_old + w_pix * float(tsdf_obs[j])) / w_new
                        self.tsdf[iz, iy, ix] = np.float32(t_new)
                        self.weight[iz, iy, ix] = np.float32(w_new)
                        self.variance[iz, iy, ix] = np.float32(1.0 / (w_new + 1e-6))
                        self.obs_count[iz, iy, ix] = min(int(self.obs_count[iz, iy, ix]) + 1, 65535)

                        if rgb is not None and cfg.integrate_color:
                            col = rgb[vi_sel[j], ui_sel[j]].astype(np.float32)
                            if col.max() > 1.0:
                                col = col / 255.0
                            cw = float(self.color_weight[iz, iy, ix])
                            self.color_sum[iz, iy, ix] = (self.color_sum[iz, iy, ix] * cw + col * w_pix) / (cw + w_pix)
                            self.color_weight[iz, iy, ix] = np.float32(cw + w_pix)

                        if self.occupancy is not None:
                            self.occupancy.log_odds[iz, iy, ix] = np.clip(
                                self.occupancy.log_odds[iz, iy, ix]
                                + np.float32(self.occupancy.cfg.log_odds_occ * w_pix),
                                self.occupancy.cfg.log_odds_min,
                                self.occupancy.cfg.log_odds_max,
                            )

        self._frames.append({"view_id": view_id, "num_pixels": int((weight_map > 0).sum())})

    def extract_mesh(self) -> trimesh.Trimesh:
        mask = self.weight >= self.cfg.min_weight_for_mesh
        tsdf_masked = self.tsdf.copy()
        tsdf_masked[~mask] = 1.0  # outside valid region

        try:
            verts, faces, normals, _ = measure.marching_cubes(
                tsdf_masked,
                level=0.0,
                spacing=(self.voxel_size, self.voxel_size, self.voxel_size),
            )
        except (ValueError, RuntimeError):
            logger.warning("Marching cubes failed; returning empty mesh")
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))

        # marching_cubes returns coords in grid space starting at origin of array
        verts = verts + self.origin
        # skimage returns (z,y,x) -> convert to (x,y,z)
        verts = verts[:, [2, 1, 0]]
        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)

    def save_outputs(self, out_dir: Path) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh_raw = self.extract_mesh()
        mesh_raw.export(out_dir / "mesh_weighted_raw.ply")

        mesh_clean = mesh_raw.copy()
        if self.cfg.mesh_cleanup and len(mesh_clean.faces) > 0:
            mesh_clean.remove_duplicate_faces()
            mesh_clean.remove_degenerate_faces()
        mesh_clean.export(out_dir / "mesh_weighted_clean.ply")

        np.savez_compressed(out_dir / "tsdf_grid.npz", tsdf=self.tsdf, weight=self.weight, variance=self.variance)
        np.savez_compressed(out_dir / "weight_grid.npz", weight=self.weight, obs_count=self.obs_count)
        np.savez_compressed(
            out_dir / "confidence_volume.npz",
            weight=self.weight,
            variance=self.variance,
            min_weight=self.cfg.min_weight_for_mesh,
        )

        paths = {
            "mesh_weighted_raw": out_dir / "mesh_weighted_raw.ply",
            "mesh_weighted_clean": out_dir / "mesh_weighted_clean.ply",
        }

        if self.occupancy is not None:
            prob = self.occupancy.probability()
            np.savez_compressed(out_dir / "occupancy_grid.npz", log_odds=self.occupancy.log_odds, prob_occ=prob)
            paths["occupancy_grid"] = out_dir / "occupancy_grid.npz"

        with (out_dir / "weighted_tsdf_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self.cfg), f)
        write_json(out_dir / "frame_list.json", {"frames": self._frames})

        return paths


class SparseHashWeightedTSDF:
    """Sparse block storage — stub for large scenes."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "SparseHashWeightedTSDF is not yet implemented. Use DenseChunkedWeightedTSDF for ROBI/bin-scale scenes."
        )

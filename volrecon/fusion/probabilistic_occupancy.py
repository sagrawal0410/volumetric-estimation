"""Probabilistic occupancy / log-odds fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OccupancyConfig:
    log_odds_free: float = -0.4
    log_odds_occ: float = 0.85
    log_odds_min: float = -10.0
    log_odds_max: float = 10.0
    surface_band_m: float = 0.01


class LogOddsOccupancyGrid:
    def __init__(self, shape: tuple[int, int, int], cfg: OccupancyConfig | None = None) -> None:
        self.shape = shape
        self.cfg = cfg or OccupancyConfig()
        self.log_odds = np.zeros(shape, dtype=np.float32)

    def update_ray(
        self,
        voxel_indices: np.ndarray,
        depths_along_ray: np.ndarray,
        surface_depth: float,
        weight: float,
    ) -> None:
        """Update voxels along a ray weighted by confidence."""
        cfg = self.cfg
        for idx, z in zip(voxel_indices, depths_along_ray, strict=True):
            i, j, k = int(idx[0]), int(idx[1]), int(idx[2])
            if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1] and 0 <= k < self.shape[2]):
                continue
            if z < surface_depth - cfg.surface_band_m:
                delta = cfg.log_odds_free * weight
            elif abs(z - surface_depth) <= cfg.surface_band_m:
                delta = cfg.log_odds_occ * weight
            else:
                continue
            self.log_odds[i, j, k] = np.clip(
                self.log_odds[i, j, k] + delta,
                cfg.log_odds_min,
                cfg.log_odds_max,
            )

    def probability(self) -> np.ndarray:
        lo = self.log_odds
        return 1.0 - 1.0 / (1.0 + np.exp(lo))

    def occupied_volume_m3(self, voxel_size_m: float, threshold: float = 0.5) -> float:
        prob = self.probability()
        count = int((prob > threshold).sum())
        return count * (voxel_size_m**3)

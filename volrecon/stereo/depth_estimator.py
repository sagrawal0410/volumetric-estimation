"""Depth estimator interface and test backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.stereo.foundation_stereo_wrapper import FoundationStereoConfig, FoundationStereoWrapper
from volrecon.stereo.types import DepthPrediction


class DepthEstimator(ABC):
    @abstractmethod
    def predict_view(self, view: ViewRecord, left_path: Path, right_path: Path, out_dir: Path) -> DepthPrediction:
        raise NotImplementedError


class FoundationStereoDepthEstimator(DepthEstimator):
    def __init__(self, cfg: FoundationStereoConfig) -> None:
        self.wrapper = FoundationStereoWrapper(cfg)

    def predict_view(self, view: ViewRecord, left_path: Path, right_path: Path, out_dir: Path) -> DepthPrediction:
        return self.wrapper.run_view(view, left_path, right_path, out_dir)


class PerfectDepthEstimator(DepthEstimator):
    """Test backend: reads depth from a provided perfect-depth npy (never GT depth path)."""

    def __init__(self, perfect_depth_paths: dict[tuple[str, str], Path] | None = None) -> None:
        self.perfect_depth_paths = perfect_depth_paths or {}

    def predict_view(self, view: ViewRecord, left_path: Path, right_path: Path, out_dir: Path) -> DepthPrediction:
        key = (view.scene_id, view.view_id)
        if key not in self.perfect_depth_paths:
            raise FileNotFoundError(f"No perfect depth registered for {key}")
        depth_m = np.load(self.perfect_depth_paths[key]).astype(np.float64)
        K = np.asarray(view.K, dtype=np.float64)
        baseline = view.stereo.baseline_m if view.stereo else 0.06
        fx = K[0, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            disparity = fx * baseline / np.maximum(depth_m, 1e-6)
        disparity[depth_m <= 0] = 0.0
        valid = depth_m > 0
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "depth_m.npy", depth_m.astype(np.float32))
        np.save(out_dir / "disparity.npy", disparity.astype(np.float32))
        return DepthPrediction(
            disparity_px=disparity,
            depth_m=depth_m,
            valid_mask=valid,
            K=K,
            baseline_m=float(baseline),
            meta={"backend": "perfect_depth_test"},
        )

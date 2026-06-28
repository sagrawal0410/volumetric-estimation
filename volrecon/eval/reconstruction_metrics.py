"""Point cloud / mesh reconstruction metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree


@dataclass
class ReconstructionMetrics:
    chamfer_l1_m: float
    accuracy_m: float
    completeness_m: float
    f_scores: dict[str, float]
    outlier_fraction: dict[str, float]
    num_sample_points: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_surface_points(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts, dtype=np.float64)


def nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    tree = cKDTree(target)
    dists, _ = tree.query(source, k=1)
    return np.asarray(dists, dtype=np.float64)


def compute_reconstruction_metrics(
    pred_mesh: trimesh.Trimesh,
    gt_mesh: trimesh.Trimesh,
    num_sample_points: int = 100_000,
    thresholds_m: list[float] | None = None,
) -> ReconstructionMetrics:
    thresholds_m = thresholds_m or [0.001, 0.002, 0.005, 0.010]
    n_pred = min(num_sample_points, max(len(pred_mesh.vertices), 1000))
    n_gt = min(num_sample_points, max(len(gt_mesh.vertices), 1000))
    pred_pts = sample_surface_points(pred_mesh, n_pred)
    gt_pts = sample_surface_points(gt_mesh, n_gt)

    d_pred_to_gt = nearest_distances(pred_pts, gt_pts)
    d_gt_to_pred = nearest_distances(gt_pts, pred_pts)

    accuracy = float(d_pred_to_gt.mean())
    completeness = float(d_gt_to_pred.mean())
    chamfer = float((d_pred_to_gt.mean() + d_gt_to_pred.mean()) / 2.0)

    f_scores: dict[str, float] = {}
    outliers: dict[str, float] = {}
    for t in thresholds_m:
        key = f"{t*1000:.0f}mm"
        prec = float((d_pred_to_gt <= t).mean())
        rec = float((d_gt_to_pred <= t).mean())
        f_scores[key] = float(2 * prec * rec / max(prec + rec, 1e-12))
        outliers[key] = float((d_pred_to_gt > t).mean())

    return ReconstructionMetrics(
        chamfer_l1_m=chamfer,
        accuracy_m=accuracy,
        completeness_m=completeness,
        f_scores=f_scores,
        outlier_fraction=outliers,
        num_sample_points=num_sample_points,
    )


def compute_depth_metrics(
    pred_depth_m: np.ndarray,
    gt_depth_m: np.ndarray,
    valid_mask: np.ndarray | None = None,
    thresholds_m: list[float] | None = None,
) -> dict[str, float]:
    thresholds_m = thresholds_m or [0.01, 0.05, 0.1]
    pred = np.asarray(pred_depth_m, dtype=np.float64)
    gt = np.asarray(gt_depth_m, dtype=np.float64)
    if valid_mask is None:
        valid_mask = (pred > 0) & (gt > 0) & np.isfinite(pred) & np.isfinite(gt)
    else:
        valid_mask = valid_mask & (gt > 0) & (pred > 0)

    if not np.any(valid_mask):
        return {"valid_pixel_ratio": 0.0, "abs_rel": float("nan"), "rmse_m": float("nan"), "mae_m": float("nan")}

    p = pred[valid_mask]
    g = gt[valid_mask]
    abs_rel = float(np.mean(np.abs(p - g) / np.maximum(g, 1e-6)))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    mae = float(np.mean(np.abs(p - g)))

    metrics: dict[str, float] = {
        "abs_rel": abs_rel,
        "rmse_m": rmse,
        "mae_m": mae,
        "valid_pixel_ratio": float(valid_mask.mean()),
        "depth_completeness": float((gt > 0).mean()),
    }
    for t in thresholds_m:
        metrics[f"bad_pixel_rate_{t}m"] = float((np.abs(p - g) > t).mean())
    return metrics

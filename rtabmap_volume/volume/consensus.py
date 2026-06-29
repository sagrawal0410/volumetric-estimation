"""Consensus volume selection and uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from rtabmap_volume.config import ConsensusConfig
from rtabmap_volume.volume.mesh_volume import VolumeEstimate
from rtabmap_volume.volume.uncertainty import compute_uncertainty


@dataclass
class ConsensusResult:
    final_volume_m3: float | None
    final_volume_liters: float | None
    confidence: str
    confidence_score_0_1: float
    recommended_estimator: str
    all_estimates: dict[str, dict]
    upper_bound_m3: float | None
    lower_bound_m3: float | None
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _est_value(estimates: dict[str, VolumeEstimate], name: str) -> float | None:
    e = estimates.get(name)
    return e.value_m3 if e and e.value_m3 is not None else None


def _robust_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def compute_consensus(
    estimates: dict[str, VolumeEstimate],
    cfg: ConsensusConfig,
    context: dict[str, Any] | None = None,
) -> ConsensusResult:
    context = context or {}
    warnings: list[str] = []
    if cfg.pile_warning:
        warnings.append(cfg.pile_warning)
    if cfg.pile_mode:
        warnings.append(
            "This estimates visible/bulk envelope volume from the RTAB-Map reconstruction, "
            "not true hidden material volume."
        )

    all_dict = {k: v.to_dict() for k, v in estimates.items()}
    upper = _est_value(estimates, "convex_hull_volume")

    # Priority-based selection
    final: float | None = None
    recommended = "none"
    confidence = "low"
    confidence_score = 0.3

    for name in cfg.estimator_priority:
        est = estimates.get(name)
        if est is None or est.value_m3 is None:
            continue
        if name in ("direct_mesh_volume", "repaired_mesh_volume") and est.reliable:
            final = est.value_m3
            recommended = name
            confidence = "high"
            confidence_score = 0.85
            break
        if cfg.pile_mode and name == "heightfield_volume":
            final = est.value_m3
            recommended = name
            confidence = "medium"
            confidence_score = 0.65
            break
        if name == "voxel_occupancy_volume" and est.value_m3 is not None:
            hull_v = _est_value(estimates, "convex_hull_volume")
            if hull_v and est.value_m3 < 0.25 * hull_v:
                continue  # surface-shell voxel artifact
            if final is None:
                final = est.value_m3
                recommended = name
                confidence = "medium" if est.reliable else "low"
                confidence_score = 0.55 if est.reliable else 0.4

    if final is None:
        robust_names = [
            "voxel_occupancy_volume",
            "alpha_shape_volume",
            "poisson_mesh_volume",
            "repaired_mesh_volume",
            "ball_pivoting_mesh_volume",
        ]
        vals = []
        hull_v = _est_value(estimates, "convex_hull_volume")
        for n in robust_names:
            v = _est_value(estimates, n)
            if v is None or v <= 0:
                continue
            if n == "voxel_occupancy_volume" and hull_v and v < 0.25 * hull_v:
                continue
            vals.append(v)
        final = _robust_median(vals)
        if final is not None:
            recommended = "robust_median"
            confidence = "low"
            confidence_score = 0.35
            warnings.append("Using median of robust estimators due to non-watertight geometry")

    unc = compute_uncertainty(estimates, context)
    confidence_score = min(confidence_score, unc["confidence_score_0_1"])
    if unc["confidence_label"] == "low":
        confidence = "low"
    warnings.extend(unc.get("warnings", []))

    liters = final * 1000.0 if final is not None else None
    lower = None
    voxel_v = _est_value(estimates, "voxel_occupancy_volume")
    if final and voxel_v:
        lower = min(final, voxel_v)

    return ConsensusResult(
        final_volume_m3=final,
        final_volume_liters=liters,
        confidence=confidence,
        confidence_score_0_1=confidence_score,
        recommended_estimator=recommended,
        all_estimates=all_dict,
        upper_bound_m3=upper,
        lower_bound_m3=lower,
        warnings=warnings,
    )

"""Uncertainty and confidence scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

from rtabmap_volume.volume.mesh_volume import VolumeEstimate


def compute_uncertainty(
    estimates: dict[str, VolumeEstimate],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    warnings: list[str] = []

    values = [e.value_m3 for e in estimates.values() if e.value_m3 is not None and e.value_m3 > 0]
    spread = 0.0
    if len(values) >= 2:
        spread = float(np.std(values) / (np.median(values) + 1e-12))
        if spread > 0.25:
            warnings.append(f"High estimator spread (CV={spread:.2f})")

    score = 0.7
    # Mesh quality penalty
    if not context.get("mesh_watertight", False):
        score -= 0.15
    if context.get("boundary_edges", 0) > 100:
        score -= 0.1
    # Segmentation penalty
    if context.get("segmentation_ambiguous", False):
        score -= 0.1
        warnings.append("Segmentation may be ambiguous")
    # Scale penalty
    if context.get("scale_warnings"):
        score -= 0.15
        warnings.extend(context["scale_warnings"])
    # Voxel convergence
    voxel = estimates.get("voxel_occupancy_volume")
    if voxel and voxel.metadata and "by_voxel_size" in (voxel.metadata or {}):
        vs = list(voxel.metadata["by_voxel_size"].values())
        if len(vs) >= 2:
            conv = (max(vs) - min(vs)) / (np.median(vs) + 1e-12)
            if conv > 0.15:
                score -= 0.1
                warnings.append("Voxel volume not converged across resolutions")

    score -= min(spread, 0.3)
    score = float(np.clip(score, 0.05, 0.95))

    label = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
    return {
        "confidence_score_0_1": score,
        "confidence_label": label,
        "estimator_spread": spread,
        "warnings": warnings,
    }

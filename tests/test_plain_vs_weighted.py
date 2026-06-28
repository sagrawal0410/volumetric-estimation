"""Synthetic plain vs weighted comparison sanity test."""

from __future__ import annotations

import numpy as np

from volrecon.fusion.weighted_tsdf import DenseChunkedWeightedTSDF, WeightedTSDFConfig


def test_low_weight_corruption_perturbs_tsdf_less():
    """Weighted fusion: low-confidence corrupted view perturbs TSDF less than high-weight corruption."""
    bounds = np.array([[0, 0, 0.9], [0.3, 0.3, 1.1]])
    K = np.array([[250, 0, 80], [0, 250, 60], [0, 0, 1]], dtype=np.float64)
    T_cw = np.eye(4)
    depth = np.full((100, 120), 1.0, dtype=np.float64)
    depth_bad = depth + 0.03  # small bias within sdf_trunc

    baseline = DenseChunkedWeightedTSDF(bounds, WeightedTSDFConfig(voxel_length_m=0.015, sdf_trunc_m=0.05, chunk_size=16))
    baseline.integrate_view(depth, np.ones_like(depth, dtype=np.float32) * 3.0, K, T_cw)

    low_w = DenseChunkedWeightedTSDF(bounds, WeightedTSDFConfig(voxel_length_m=0.015, sdf_trunc_m=0.05, chunk_size=16))
    low_w.integrate_view(depth, np.ones_like(depth, dtype=np.float32) * 3.0, K, T_cw)
    low_w.integrate_view(depth_bad, np.ones_like(depth_bad, dtype=np.float32) * 0.1, K, T_cw)

    high_w = DenseChunkedWeightedTSDF(bounds, WeightedTSDFConfig(voxel_length_m=0.015, sdf_trunc_m=0.05, chunk_size=16))
    high_w.integrate_view(depth, np.ones_like(depth, dtype=np.float32) * 3.0, K, T_cw)
    high_w.integrate_view(depth_bad, np.ones_like(depth_bad, dtype=np.float32) * 3.0, K, T_cw)

    mask = baseline.weight > 0.1
    assert mask.any()
    delta_low = float(np.abs(low_w.tsdf - baseline.tsdf)[mask].sum())
    delta_high = float(np.abs(high_w.tsdf - baseline.tsdf)[mask].sum())
    assert delta_high > delta_low > 0.0

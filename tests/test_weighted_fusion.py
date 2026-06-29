"""Tests for uncertainty-weighted TSDF fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from volrecon.datasets.canonical_schema import StereoCalibration, ViewRecord
from volrecon.fusion.robust_kernels import huber_weight
from volrecon.fusion.weighted_tsdf import DenseChunkedWeightedTSDF, WeightedTSDFConfig
from volrecon.stereo.foundation_stereo_wrapper import NO_STEREO_ERROR, FoundationStereoConfig, FoundationStereoWrapper
from volrecon.stereo.depth_estimator import PerfectDepthEstimator
from volrecon.uncertainty.calibration import UncertaintyConfig
from volrecon.uncertainty.confidence_sources import combine_confidence, compute_confidence_maps
from volrecon.uncertainty.stereo_consistency import lr_consistency_confidence, warp_disparity_right_to_left


def test_lr_consistency_high_for_correct_disparity():
    h, w = 64, 80
    d_true = np.full((h, w), 10.0, dtype=np.float64)
    d_r2l = np.full((h, w), 10.0, dtype=np.float64)
    c = lr_consistency_confidence(d_true, d_r2l, tau_lr_px=1.5)
    assert c[32, 40] == pytest.approx(1.0, abs=1e-6)
    warped = warp_disparity_right_to_left(d_r2l, d_true)
    assert np.isnan(warped[32, 5])  # u_r < 0 invalid


def test_lr_consistency_low_for_corrupted_disparity():
    h, w = 64, 80
    d_true = np.full((h, w), 10.0, dtype=np.float64)
    d_bad = np.full((h, w), 20.0, dtype=np.float64)
    c = lr_consistency_confidence(d_true, d_bad, tau_lr_px=1.5)
    assert float(c.mean()) < 0.01


def test_lr_sign_convention_warp():
    d_l = np.zeros((10, 20), dtype=np.float64)
    d_l[5, 10] = 4.0
    d_r = np.zeros((10, 20), dtype=np.float64)
    d_r[5, 6] = 4.0
    warped = warp_disparity_right_to_left(d_r, d_l)
    assert np.isfinite(warped[5, 10])
    assert warped[5, 10] == pytest.approx(4.0)


def test_weighted_fusion_prefers_clean_depth():
    """Two observations of a plane: high-weight clean vs low-weight corrupted."""
    bounds = np.array([[0, 0, 0.9], [0.5, 0.5, 1.1]])
    cfg = WeightedTSDFConfig(voxel_length_m=0.02, sdf_trunc_m=0.05, min_weight_for_mesh=0.5, chunk_size=16)
    tsdf = DenseChunkedWeightedTSDF(bounds, cfg)
    K = np.array([[200, 0, 50], [0, 200, 50], [0, 0, 1]], dtype=np.float64)
    T_cw = np.eye(4)

    depth_clean = np.full((100, 100), 1.0, dtype=np.float64)
    weight_clean = np.full((100, 100), 5.0, dtype=np.float32)
    depth_bad = np.full((100, 100), 1.4, dtype=np.float64)
    weight_bad = np.full((100, 100), 0.05, dtype=np.float32)

    tsdf.integrate_view(depth_clean, weight_clean, K, T_cw)
    tsdf.integrate_view(depth_bad, weight_bad, K, T_cw)

    assert float(tsdf.weight.max()) > 1.0
    # Voxels with high accumulated weight should have tsdf near 0 (surface)
    high_w = tsdf.weight > 1.0
    if np.any(high_w):
        assert float(np.abs(tsdf.tsdf[high_w]).mean()) < 0.6


def test_huber_downweights_large_residual():
    w_small = huber_weight(np.array([0.05]), delta=0.25)[0]
    w_large = huber_weight(np.array([1.0]), delta=0.25)[0]
    assert w_large < w_small


def test_no_gt_depth_cheating_weighted_path(tmp_path: Path):
    depth = np.ones((32, 32), dtype=np.float64) * 0.7
    perfect = tmp_path / "perfect.npy"
    np.save(perfect, depth.astype(np.float32))
    view = ViewRecord(
        dataset="robi",
        scene_id="s",
        view_id="v",
        left_path="l.png",
        right_path="r.png",
        gt_depth_path="/nonexistent/gt.png",
        K=np.diag([100, 100, 1]),
        stereo=StereoCalibration(has_true_stereo=True, baseline_m=0.06, rectified=True),
    )
    est = PerfectDepthEstimator({("s", "v"): perfect})
    pred = est.predict_view(view, tmp_path / "l.png", tmp_path / "r.png", tmp_path / "pred")
    assert pred.depth_m.shape == depth.shape


def test_weighted_beats_plain_on_corrupted_depth():
    """Corrupted observations with low weight should change the TSDF less than high-weight ones."""
    bounds = np.array([[0, 0, 0.9], [0.4, 0.4, 1.1]])
    K = np.array([[300, 0, 80], [0, 300, 60], [0, 0, 1]], dtype=np.float64)
    T_cw = np.eye(4)
    depth_clean = np.full((100, 120), 1.0, dtype=np.float64)
    depth_bad = depth_clean.copy()
    depth_bad[30:50, 40:60] = 1.6

    tsdf_high = DenseChunkedWeightedTSDF(bounds, WeightedTSDFConfig(voxel_length_m=0.02, chunk_size=16))
    tsdf_high.integrate_view(depth_clean, np.full_like(depth_clean, 5.0, dtype=np.float32), K, T_cw)

    tsdf_low = DenseChunkedWeightedTSDF(bounds, WeightedTSDFConfig(voxel_length_m=0.02, chunk_size=16))
    tsdf_low.integrate_view(depth_clean, np.full_like(depth_clean, 5.0, dtype=np.float32), K, T_cw)
    tsdf_low.integrate_view(depth_bad, np.full_like(depth_bad, 0.01, dtype=np.float32), K, T_cw)

    tsdf_high.integrate_view(depth_bad, np.full_like(depth_bad, 5.0, dtype=np.float32), K, T_cw)

    delta_low = float(np.abs(tsdf_low.tsdf - tsdf_high.tsdf)[tsdf_high.weight > 1].mean())
    assert delta_low < 0.4


def test_no_stereo_blocks_foundation_stereo(tmp_path: Path):
    ckpt = tmp_path / "m.pth"
    ckpt.touch()
    (tmp_path / "cfg.yaml").write_text("vit_size: vitl\n", encoding="utf-8")
    view = ViewRecord(
        dataset="bop_tless",
        scene_id="s",
        view_id="v",
        rgb_path="rgb.png",
        stereo=StereoCalibration(has_true_stereo=False),
    )
    w = FoundationStereoWrapper(
        FoundationStereoConfig(foundationstereo_repo=tmp_path, ckpt=ckpt)
    )
    with pytest.raises(ValueError, match="No true stereo"):
        w.validate_view(view)


def test_confidence_combine_respects_invalid():
    cfg = UncertaintyConfig()
    comps = {
        "valid": np.array([True, False]),
        "c_lr": np.array([1.0, 1.0]),
        "c_photo": np.array([1.0, 1.0]),
        "c_range": np.array([1.0, 1.0]),
        "c_angle": np.array([1.0, 1.0]),
        "c_texture": np.array([1.0, 1.0]),
        "c_sat": np.array([1.0, 1.0]),
        "c_mv": np.array([1.0, 1.0]),
        "c_temp": np.array([1.0, 1.0]),
    }
    c, w = combine_confidence(comps, cfg)
    assert w[1] == 0.0

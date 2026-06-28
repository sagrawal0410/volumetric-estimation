"""Synthetic plain vs weighted comparison sanity test."""

from __future__ import annotations

import numpy as np

from volrecon.fusion.open3d_tsdf import PlainTSDFConfig, PlainTSDFReconstructor
from volrecon.fusion.weighted_tsdf import DenseChunkedWeightedTSDF, WeightedTSDFConfig
from volrecon.eval.reconstruction_metrics import compute_reconstruction_metrics
import trimesh


def test_weighted_lower_chamfer_than_plain_on_synthetic_outliers(tmp_path):
    """On synthetic corrupted depth, weighted fusion should beat plain when mesh extracts."""
    bounds = np.array([[0, 0, 0.85], [0.3, 0.3, 1.15]])
    K = np.array([[250, 0, 80], [0, 250, 60], [0, 0, 1]], dtype=np.float64)
    T_cw = np.eye(4)

    depth = np.full((120, 160), 1.0, dtype=np.float64)
    depth[50:70, 70:90] = 1.5  # outlier bump
    weight = np.ones((120, 160), dtype=np.float32) * 2.0
    weight[50:70, 70:90] = 0.01

    np.save(tmp_path / "d.npy", depth.astype(np.float32))
    plain = PlainTSDFReconstructor(
        PlainTSDFConfig(voxel_length_m=0.01, sdf_trunc_m=0.03, integrate_color=False),
        bounds,
    )
    plain.integrate_view(None, tmp_path / "d.npy", K, T_cam_world=T_cw)
    plain_mesh = trimesh.Trimesh(
        vertices=np.asarray(plain.extract_mesh().vertices),
        faces=np.asarray(plain.extract_mesh().triangles),
        process=False,
    )

    weighted = DenseChunkedWeightedTSDF(
        bounds,
        WeightedTSDFConfig(voxel_length_m=0.01, sdf_trunc_m=0.03, min_weight_for_mesh=0.5, chunk_size=16),
    )
    weighted.integrate_view(depth, weight, K, T_cw)
    w_mesh = weighted.extract_mesh()

    gt = trimesh.creation.box(extents=(0.25, 0.25, 0.05))
    gt.apply_translation([0.12, 0.12, 1.0])

    if len(plain_mesh.vertices) > 50 and len(w_mesh.vertices) > 50:
        m_plain = compute_reconstruction_metrics(plain_mesh, gt, num_sample_points=3000)
        m_weighted = compute_reconstruction_metrics(w_mesh, gt, num_sample_points=3000)
        assert m_weighted.chamfer_l1_m <= m_plain.chamfer_l1_m + 0.05

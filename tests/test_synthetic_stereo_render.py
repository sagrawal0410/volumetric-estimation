"""Tests for synthetic stereo rendering."""

from __future__ import annotations

import numpy as np
import trimesh

from volrecon.geometry.render_gt import render_synthetic_stereo_pair


def test_synthetic_stereo_images_differ_for_nonzero_baseline():
    mesh = trimesh.creation.icosphere(radius=0.1)
    mesh.apply_translation([0, 0, 0.8])
    K = np.array([[500.0, 0, 160], [0, 500.0, 120], [0, 0, 1]])
    w, h = 320, 240
    baseline = 0.06

    out = render_synthetic_stereo_pair([mesh], [np.eye(4)], K, w, h, baseline)
    left = out["left_rgb"]
    right = out["right_rgb"]
    left_d = out["left_depth_m"]
    right_d = out["right_depth_m"]

    assert out["synthetic"] is True
    assert not np.array_equal(left, right)
    assert not np.allclose(left_d, right_d)
    assert out["baseline_m"] == baseline


def test_synthetic_stereo_metadata_flag():
    mesh = trimesh.creation.icosphere(radius=0.1)
    mesh.apply_translation([0, 0, 0.8])
    K = np.array([[400.0, 0, 100], [0, 400.0, 100], [0, 0, 1]])
    out = render_synthetic_stereo_pair([mesh], [np.eye(4)], K, 200, 200, 0.06)
    assert out["synthetic"] is True
    assert "T_left_right" in out

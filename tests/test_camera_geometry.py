"""Tests for camera geometry."""

from __future__ import annotations

import numpy as np

from volrecon.geometry.camera import (
    backproject_depth,
    depth_to_pointcloud,
    disparity_to_depth,
    project_points,
    resize_intrinsics,
)
from volrecon.geometry.transforms import invert_T, make_T, transform_points


def test_backproject_project_roundtrip():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    depth = np.zeros((480, 640), dtype=np.float64)
    depth[240, 320] = 2.0
    pts = backproject_depth(depth, K)
    assert pts.shape == (1, 3)
    assert np.allclose(pts[0, 2], 2.0)
    uv, z = project_points(pts, K)
    assert np.allclose(uv[0], [320, 240], atol=1e-5)
    assert np.allclose(z[0], 2.0)


def test_disparity_to_depth_scalar():
    depth = disparity_to_depth(100.0, fx_px=1000.0, baseline_m=0.1)
    assert np.isclose(depth, 1.0)


def test_resize_intrinsics():
    K = np.eye(3)
    K[0, 0] = K[1, 1] = 100.0
    K[0, 2] = 50.0
    K[1, 2] = 40.0
    K2 = resize_intrinsics(K, 2.0, 2.0)
    assert K2[0, 0] == 200.0
    assert K2[0, 2] == 100.0


def test_transform_roundtrip():
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    T = make_T(R, t)
    p = np.array([[0.0, 0.0, 0.0]])
    p2 = transform_points(T, p)
    p3 = transform_points(invert_T(T), p2)
    assert np.allclose(p, p3, atol=1e-9)

"""Tests for camera geometry utilities."""

import numpy as np
import pytest

from volume_benchmark.common.geometry import (
    backproject_depth_to_object,
    convert_mm_pose_to_meters,
    invert_T,
    make_T,
    project_object_points_to_camera,
    transform_points,
)


def test_make_T_invert_T_roundtrip():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    t = np.array([0.1, -0.2, 1.5])
    T = make_T(R, t)
    T_inv = invert_T(T)
    assert np.allclose(T @ T_inv, np.eye(4), atol=1e-10)


def test_transform_points():
    T = make_T(np.eye(3), np.array([1.0, 0.0, 0.0]))
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = transform_points(pts, T)
    assert np.allclose(out, [[1, 0, 0], [2, 1, 1]])


def test_convert_mm_pose_to_meters():
    T = np.eye(4)
    T[:3, 3] = [1000.0, 2000.0, 500.0]
    T_m = convert_mm_pose_to_meters(T)
    assert np.allclose(T_m[:3, 3], [1.0, 2.0, 0.5])


def test_backproject_and_project_roundtrip():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    T = make_T(np.eye(3), np.array([0.0, 0.0, 1.0]))
    depth = np.zeros((480, 640), dtype=np.float32)
    mask = np.zeros((480, 640), dtype=bool)
    depth[240, 320] = 1.0
    mask[240, 320] = True

    pts_obj = backproject_depth_to_object(depth, mask, K, T)
    assert pts_obj.shape == (1, 3)
    u, v, z = project_object_points_to_camera(pts_obj, K, invert_T(T), (480, 640))
    assert u.size == 1
    assert abs(u[0] - 320) < 1e-3
    assert abs(v[0] - 240) < 1e-3
    assert abs(z[0] - 1.0) < 1e-3


def test_backproject_depth_bounds():
    K = np.eye(3)
    K[0, 0] = K[1, 1] = 500
    K[0, 2], K[1, 2] = 10, 10
    depth = np.array([[0.01, 10.0], [1.0, 1.0]], dtype=np.float32)
    mask = np.ones_like(depth, dtype=bool)
    T = np.eye(4)
    pts = backproject_depth_to_object(depth, mask, K, T, depth_min=0.05, depth_max=5.0)
    assert pts.shape[0] == 2  # only center pixel at 1m in each row... actually 4 pixels, 2 valid

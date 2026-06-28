"""Unit tests for BOP depth and pose conversions."""

from __future__ import annotations

import numpy as np

from tless_volume_benchmark.geometry import bop_pose_m2c_to_T_cam_to_object


def test_depth_mm_to_meters():
    raw = np.array([[0, 1000], [500, 0]], dtype=np.uint16)
    depth_m = raw.astype(np.float32) * 1.0 / 1000.0
    depth_m[raw == 0] = 0.0
    assert depth_m[0, 1] == 1.0
    assert depth_m[1, 0] == 0.5
    assert depth_m[1, 1] == 0.0


def test_cam_t_mm_to_meters_in_pose():
    R = np.eye(3)
    t_mm = [0.0, 0.0, 1000.0]
    T = bop_pose_m2c_to_T_cam_to_object(R, t_mm)
    # T_cam_to_object = inverse(T_m2c); camera at z=1m in object frame
    cam_center = T[:3, 3]
    assert np.allclose(cam_center, [0.0, 0.0, -1.0], atol=1e-6)

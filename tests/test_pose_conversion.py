"""Tests for BOP pose to object-centric transform."""

from __future__ import annotations

import numpy as np

from tless_volume_benchmark.geometry import (
    bop_pose_m2c_to_T_cam_to_object,
    invert_T,
    make_T,
    transform_points,
)


def test_identity_rotation_translation_1m():
    R = np.eye(3)
    t_mm = np.array([0.0, 0.0, 1000.0])
    T_cam_to_object = bop_pose_m2c_to_T_cam_to_object(R, t_mm)
    T_m2c = invert_T(T_cam_to_object)

    origin_obj = np.array([[0.0, 0.0, 0.0]])
    origin_cam = transform_points(origin_obj, T_m2c)
    assert np.allclose(origin_cam[0], [0.0, 0.0, 1.0], atol=1e-6)

    cam_center_obj = T_cam_to_object[:3, 3]
    assert np.allclose(cam_center_obj, [0.0, 0.0, -1.0], atol=1e-6)


def test_roundtrip_model_point():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    t_mm = np.array([100.0, 200.0, 800.0])
    T_cam_to_object = bop_pose_m2c_to_T_cam_to_object(R, t_mm)
    T_m2c = make_T(R, t_mm / 1000.0)
    assert np.allclose(T_m2c, invert_T(T_cam_to_object), atol=1e-9)

    p_obj = np.array([[0.05, -0.02, 0.01]])
    p_cam = transform_points(p_obj, T_m2c)
    p_back = transform_points(p_cam, T_cam_to_object)
    assert np.allclose(p_obj, p_back, atol=1e-9)

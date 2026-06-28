"""Tests for BOP pose conventions."""

from __future__ import annotations

import numpy as np
import trimesh

from volrecon.geometry.transforms import bop_T_cam_model_to_meters, bop_T_model_cam_to_meters, transform_points


def test_bop_pose_inverse_maps_camera_to_model():
    # Cube centered at origin in model frame
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    corner = np.asarray(mesh.vertices[0])

    R_m2c = np.eye(3).flatten().tolist()
    t_m2c_mm = [0.0, 0.0, 500.0]  # camera 0.5 m along +Z from model origin

    T_model_cam = bop_T_model_cam_to_meters(R_m2c, t_m2c_mm)
    T_cam_model = bop_T_cam_model_to_meters(R_m2c, t_m2c_mm)

    # Transform corner to camera frame
    p_cam = transform_points(T_model_cam, corner.reshape(1, 3))
    # Back to model frame
    p_model = transform_points(T_cam_model, p_cam)
    assert np.allclose(corner, p_model[0], atol=1e-9)

    # Known: point at model origin maps to z=0.5 in camera frame (t converted to meters)
    origin_cam = transform_points(T_model_cam, np.zeros((1, 3)))
    assert np.allclose(origin_cam[0, 2], 0.5, atol=1e-9)

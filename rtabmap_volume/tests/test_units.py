"""Unit scaling tests."""

import numpy as np
import trimesh

from rtabmap_volume.preprocess.scale_units import apply_scaling, infer_scale_warnings, scale_mesh_to_meters


def test_mm_cube_converts_to_m():
    cube_mm = trimesh.creation.box(extents=[1000, 1000, 1000])
    cube_m = scale_mesh_to_meters(cube_mm, "mm")
    est_vol = cube_m.volume
    assert abs(est_vol - 1.0) < 0.05


def test_suspicious_scale_warning():
    warnings = infer_scale_warnings(1500.0)
    assert any("millimeter" in w.lower() for w in warnings)


def test_tiny_scale_warning():
    warnings = infer_scale_warnings(0.0005)
    assert len(warnings) > 0

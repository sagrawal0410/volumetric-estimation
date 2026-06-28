"""Unit tests for disparity -> depth conversion."""

import numpy as np

from volume_benchmark.stereo.disparity_depth import disparity_to_depth_m, make_depth_valid_mask


def test_disparity_to_depth_known_values():
    disp = np.array([[84.0]], dtype=np.float32)
    depth = disparity_to_depth_m(disp, fx_px=700.0, baseline_m=0.12)
    assert np.isclose(depth[0, 0], 1.0, rtol=1e-5)


def test_disparity_to_depth_invalid_low_disp():
    disp = np.array([[0.05, 84.0]], dtype=np.float32)
    depth = disparity_to_depth_m(disp, fx_px=700.0, baseline_m=0.12, min_disp=0.1)
    assert depth[0, 0] == 0.0
    assert np.isclose(depth[0, 1], 1.0, rtol=1e-5)


def test_disparity_to_depth_max_depth_clip():
    disp = np.array([[1.0]], dtype=np.float32)
    depth = disparity_to_depth_m(disp, fx_px=700.0, baseline_m=0.12, max_depth_m=5.0)
    assert depth[0, 0] == 0.0


def test_make_depth_valid_mask_with_object_mask():
    depth = np.array([[0.5, 0.0], [2.0, 0.3]], dtype=np.float32)
    mask = np.array([[True, True], [False, True]])
    valid = make_depth_valid_mask(depth, object_mask=mask)
    assert valid[0, 0]
    assert not valid[0, 1]
    assert not valid[1, 0]
    assert valid[1, 1]

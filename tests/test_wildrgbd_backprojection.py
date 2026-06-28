"""Tests for WildRGB-D backprojection."""

import numpy as np

from wildrgbd_volume_benchmark.geometry import backproject_depth


def test_flat_depth_plane():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    depth_m = np.full((480, 640), 1.0, dtype=np.float32)
    mask = np.zeros((480, 640), dtype=bool)
    mask[200:280, 280:360] = True
    T = np.eye(4)
    pts = backproject_depth(depth_m, mask, K, T)
    assert pts.shape[0] > 0
    assert np.allclose(pts[:, 2], 1.0, atol=1e-3)
    cx, cy = 320, 240
    center = pts.mean(axis=0)
    assert abs(center[0]) < 0.05
    assert abs(center[1]) < 0.05

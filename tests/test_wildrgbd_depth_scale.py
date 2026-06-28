"""Tests for WildRGB-D depth scale."""

import numpy as np

from wildrgbd_volume_benchmark.io_wildrgbd import load_depth_m


def test_depth_1000_is_one_meter(tmp_path):
    import cv2

    raw = np.array([[0, 1000], [500, 0]], dtype=np.uint16)
    path = tmp_path / "depth.png"
    cv2.imwrite(str(path), raw)
    depth_m = load_depth_m(path)
    assert depth_m[0, 1] == 1.0
    assert depth_m[1, 0] == 0.5
    assert depth_m[1, 1] == 0.0

"""Tests for WildRGB-D pose loading."""

import numpy as np

from wildrgbd_volume_benchmark.io_wildrgbd import load_cam_poses


def test_identity_pose_parsing(tmp_path):
    T = np.eye(4)
    line = "000001 " + " ".join(f"{x:.6f}" for x in T.reshape(-1))
    path = tmp_path / "cam_poses.txt"
    path.write_text(line + "\n", encoding="utf-8")
    poses = load_cam_poses(tmp_path)
    assert "000001" in poses
    assert np.allclose(poses["000001"], np.eye(4))

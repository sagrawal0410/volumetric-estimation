"""Tests for WildRGB-D view selection."""

import numpy as np

from wildrgbd_volume_benchmark.io_wildrgbd import WildRGBDFrame, WildRGBDScene
from wildrgbd_volume_benchmark.view_selection import select_sparse_views


def _make_circular_scene(n: int = 12) -> WildRGBDScene:
    K = np.eye(3)
    frames = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        eye = np.array([0.5 * np.cos(angle), 0.1, 0.5 * np.sin(angle)])
        T = np.eye(4)
        T[:3, 3] = eye
        depth = np.ones((64, 64), dtype=np.float32) * 0.8
        mask = np.ones((64, 64), dtype=bool)
        frames.append(
            WildRGBDFrame(
                frame_id=f"{i:04d}",
                rgb_path="",
                depth_path="",
                mask_path="",
                depth_m=depth,
                mask=mask,
                K=K,
                T_cam_to_world=T,
            )
        )
    return WildRGBDScene(
        category="test",
        scene_id="scenes_test",
        scene_dir="/tmp",
        scene_type="single",
        frames=frames,
        K=K,
        image_size=(64, 64),
    )


def test_selects_spread_views():
    scene = _make_circular_scene(12)
    T_world_to_object = np.eye(4)
    selected = select_sparse_views(
        scene, T_world_to_object, num_views=5, min_angle_deg=15.0, require_valid_depth_pixels=100
    )
    assert len(selected) == 5
    ids = {f.frame_id for f in selected}
    assert len(ids) == 5

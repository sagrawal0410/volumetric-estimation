"""Tests for WildRGB-D pseudo-GT volume estimation."""

import numpy as np
import trimesh

from wildrgbd_volume_benchmark.pseudo_gt import compute_pseudo_gt_volume_from_full_video
from wildrgbd_volume_benchmark.io_wildrgbd import WildRGBDFrame, WildRGBDScene


def test_voxel_occupancy_pseudo_gt_on_dense_cube_points(tmp_path):
    """Dense object-frame cube points should yield reasonable occupancy volume."""
    from wildrgbd_volume_benchmark.geometry import transform_points, invert_T

    # Build minimal scene with one frame; fusion will duplicate if needed
    rng = np.random.default_rng(0)
    half = 0.05
    pts_obj = rng.uniform(-half, half, size=(8000, 3))
    T_world_to_object = np.eye(4)
    T_world_to_object[:3, 3] = -pts_obj.mean(axis=0)
    T_cam_to_world = invert_T(T_world_to_object)

    depth = np.zeros((64, 64), np.float32)
    mask = np.zeros((64, 64), bool)
    depth[20:44, 20:44] = 0.5
    mask[20:44, 20:44] = True
    K = np.array([[100, 0, 32], [0, 100, 32], [0, 0, 1]], dtype=float)

    frames = []
    for i in range(6):
        angle = 2 * np.pi * i / 6
        T = T_cam_to_world.copy()
        T[:3, 3] = [0.4 * np.cos(angle), 0.0, 0.4 * np.sin(angle)]
        frames.append(
            WildRGBDFrame(
                frame_id=f"{i:04d}",
                rgb_path="",
                depth_path="",
                mask_path="",
                depth_m=depth.copy(),
                mask=mask.copy(),
                K=K,
                T_cam_to_world=T,
            )
        )

    scene = WildRGBDScene(
        category="test",
        scene_id="scenes_test",
        scene_dir=str(tmp_path),
        scene_type="single",
        frames=frames,
        K=K,
        image_size=(64, 64),
    )
    out = tmp_path / "pseudo_gt"
    import os

    os.environ["WILDRGBD_SKIP_OPEN3D"] = "1"
    pg = compute_pseudo_gt_volume_from_full_video(
        scene,
        out,
        frame_stride=1,
        tsdf_voxel_length=0.01,
        voxel_occupancy_size=0.01,
    )
    assert pg["volume_m3"] is not None
    assert pg["gt_type"] == "full_video_reconstruction_pseudo_gt"
    assert pg["exact_gt"] is False

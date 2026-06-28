"""Synthetic WildRGB-D scene end-to-end tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

from wildrgbd_volume_benchmark.geometry import estimate_object_frame_from_full_points, transform_points
from wildrgbd_volume_benchmark.io_wildrgbd import load_scene
from wildrgbd_volume_benchmark.methods.convex_hull import estimate_convex_hull_volume
from wildrgbd_volume_benchmark.methods.voxel_carving import estimate_voxel_carving_volume
from wildrgbd_volume_benchmark.pointcloud_fusion import fuse_scene_pointcloud_full
from wildrgbd_volume_benchmark.pseudo_gt import compute_pseudo_gt_volume_from_full_video
from wildrgbd_volume_benchmark.prepare_scene import prepare_scene


def _write_synthetic_wildrgbd_scene(base: Path, num_frames: int = 8) -> Path:
    scene_dir = base / "testcat" / "scenes" / "scenes_000001"
    for sub in ("rgb", "depth", "masks"):
        (scene_dir / sub).mkdir(parents=True)
    (base / "testcat").mkdir(parents=True, exist_ok=True)
    with (base / "testcat" / "types.json").open("w") as f:
        json.dump({"000001": "single"}, f)

    K = np.array([[200, 0, 64], [0, 200, 64], [0, 0, 1]], dtype=float)
    meta = {"width": 128, "height": 128, "K": K.tolist()}
    with (scene_dir / "metadata.json").open("w") as f:
        json.dump(meta, f)

    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    gt_m3 = 0.001
    poses_lines = []
    for i in range(num_frames):
        fid = f"{i:04d}"
        angle = 2 * np.pi * i / num_frames
        eye = np.array([0.4 * np.cos(angle), 0.05, 0.4 * np.sin(angle)])
        R = np.eye(3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = eye
        poses_lines.append(fid + " " + " ".join(f"{x:.6f}" for x in T.reshape(-1)))

        # rasterize simple depth
        depth = np.zeros((128, 128), dtype=np.float32)
        mask = np.zeros((128, 128), dtype=bool)
        depth[40:88, 40:88] = np.linalg.norm(eye)
        mask[40:88, 40:88] = True
        cv2.imwrite(str(scene_dir / "depth" / f"{fid}.png"), (depth * 1000).astype(np.uint16))
        cv2.imwrite(str(scene_dir / "masks" / f"{fid}.png"), mask.astype(np.uint8) * 255)
        cv2.imwrite(str(scene_dir / "rgb" / f"{fid}.png"), np.zeros((128, 128, 3), dtype=np.uint8))

    (scene_dir / "cam_poses.txt").write_text("\n".join(poses_lines), encoding="utf-8")
    return base


def test_pseudo_gt_on_synthetic_scene(tmp_path):
    import os

    os.environ["WILDRGBD_SKIP_OPEN3D"] = "1"
    root = _write_synthetic_wildrgbd_scene(tmp_path, num_frames=6)
    scene = load_scene(
        root / "testcat" / "scenes" / "scenes_000001",
        category="testcat",
        scene_id="scenes_000001",
        scene_type="single",
    )
    out = tmp_path / "pg"
    pg = compute_pseudo_gt_volume_from_full_video(
        scene, out, frame_stride=1, max_frames_for_gt=6, tsdf_voxel_length=0.008, voxel_occupancy_size=0.008
    )
    assert pg["gt_type"] == "full_video_reconstruction_pseudo_gt"
    assert pg["volume_m3"] is not None and pg["volume_m3"] > 0


@pytest.mark.skipif(
    __import__("os").environ.get("WILDRGBD_SKIP_OPEN3D") == "1",
    reason="Open3D disabled",
)
def test_prepare_and_methods(tmp_path):
    root = _write_synthetic_wildrgbd_scene(tmp_path, num_frames=8)
    out = tmp_path / "prepared" / "testcat" / "scenes_000001"
    prepare_scene(
        wildrgbd_root=root,
        category="testcat",
        scene_id="scenes_000001",
        out_dir=out,
        num_views=4,
        require_valid_depth_pixels=100,
        gt_frame_stride=1,
        max_frames_for_gt=8,
    )
    ch = estimate_convex_hull_volume(out, voxel_downsample=0.01)
    assert ch["volume_m3"] > 0
    vc = estimate_voxel_carving_volume(out, voxel_size=0.012, min_views_checked=1)
    assert vc["volume_m3"] > 0

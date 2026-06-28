"""Tests for BigBIRD dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh
import yaml

from volume_benchmark.common.geometry import invert_T
from volume_benchmark.common.io import load_prepared_scan, validate_prepared_scan
from volume_benchmark.common.view_selection import select_bigbird_views
from volume_benchmark.datasets.bigbird_adapter import (
    count_valid_depth_pixels,
    discover_view_candidates,
    prepare_bigbird_scan,
    rasterize_points_to_depth,
    resolve_ground_truth,
    BigBirdConfig,
)
from tests.conftest import _fill_silhouette, _look_at_pose


def _make_open_box_mesh(size: float = 0.12) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=(size, size, size))
    return trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2])


def _write_fake_bigbird_object(
    object_root: Path,
    num_views: int = 10,
    image_size: tuple[int, int] = (160, 160),
    include_merged_cloud: bool = True,
) -> tuple[trimesh.Trimesh, float]:
    """Create a fake BigBIRD object folder with mesh, views, calibration, and merged cloud."""
    object_root.mkdir(parents=True, exist_ok=True)
    full_box = trimesh.creation.box(extents=(0.12, 0.12, 0.12))
    gt_volume = 0.12 ** 3

    open_mesh = _make_open_box_mesh()
    mesh_dir = object_root / "reconstruction"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    open_mesh.export(mesh_dir / "object_mesh.ply")

    if include_merged_cloud:
        cloud_dir = object_root / "pointclouds"
        cloud_dir.mkdir(parents=True, exist_ok=True)
        surface, _ = trimesh.sample.sample_surface(full_box, 5000)
        half = 0.06
        interior = np.random.default_rng(0).uniform(-half, half, size=(15000, 3))
        merged = trimesh.PointCloud(np.vstack([surface, interior]))
        merged.export(cloud_dir / "merged_cloud.ply")

    width, height = image_size
    K = np.array(
        [[220.0, 0.0, width / 2], [0.0, 220.0, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    calib_dir = object_root / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    np.save(calib_dir / "K.npy", K)

    views_dir = object_root / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    radius = 0.55
    surface_points, _ = trimesh.sample.sample_surface(full_box, 8000)
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        eye = np.array([radius * np.cos(angle), 0.05 * (i % 2), radius * np.sin(angle)])
        T = _look_at_pose(eye)
        depth_m, mask = rasterize_points_to_depth(
            surface_points, K, T, image_shape=(height, width)
        )
        depth_m, mask = _fill_silhouette(depth_m, mask)
        if count_valid_depth_pixels(depth_m, mask) < 500:
            depth_m = np.full((height, width), 0.55, dtype=np.float32)
            mask = np.ones((height, width), dtype=bool)

        view_dir = views_dir / f"view_{i:02d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        np.save(view_dir / f"depth_{i:02d}.npy", depth_m.astype(np.float32))
        cv2.imwrite(str(view_dir / f"mask_{i:02d}.png"), (mask.astype(np.uint8) * 255))
        np.save(view_dir / f"pose_{i:02d}.npy", T)

    config = {
        "depth_glob": "depth_*.npy",
        "mask_glob": "mask_*.png",
        "pose_glob": "pose_*.npy",
        "calibration_file": "calibration/K.npy",
        "depth_scale_to_meters": 1.0,
        "pose_format": "cam_to_object",
        "mesh_units": "m",
    }
    with (object_root / "bigbird_config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    return full_box, gt_volume


def _min_selected_angle_deg(poses: list[np.ndarray], indices: list[int]) -> float:
    from volume_benchmark.common.view_selection import angular_distance_deg, view_direction_object

    directions = [view_direction_object(poses[i]) for i in indices]
    min_angle = 180.0
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            min_angle = min(min_angle, angular_distance_deg(directions[i], directions[j]))
    return min_angle


def test_select_bigbird_views_prefers_coverage_and_spread():
    poses = []
    valid_counts = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        eye = np.array([0.5 * np.cos(angle), 0.0, 0.5 * np.sin(angle)])
        poses.append(_look_at_pose(eye))
        valid_counts.append(5000 - i * 100)

    selected = select_bigbird_views(
        poses, valid_counts, num_views=5, min_valid_depth_pixels=1000
    )
    assert len(selected) == 5
    assert selected[0] == 0  # most valid pixels
    assert _min_selected_angle_deg(poses, selected) > 20.0


def test_resolve_pseudo_gt_when_mesh_not_watertight(tmp_path: Path):
    object_root = tmp_path / "object"
    _write_fake_bigbird_object(object_root, num_views=4)

    config = BigBirdConfig.from_yaml(object_root / "bigbird_config.yaml")
    out_dir = tmp_path / "gt_out"
    out_dir.mkdir()
    gt = resolve_ground_truth(
        object_root,
        config,
        out_dir,
        gt_source="mesh_then_merged_pointcloud",
        gt_voxel_size=0.002,
        repair_mesh=False,
    )
    assert gt.gt_type == "full_reconstruction_pseudo_gt"
    assert gt.watertight is False
    assert gt.exact_gt is False
    assert gt.pseudo_gt_method in ("alpha_shape", "poisson", "voxel_occupancy")
    assert gt.volume_m3 > 0


def test_prepare_bigbird_scan_end_to_end(tmp_path: Path):
    object_root = tmp_path / "advil_liqui_gels"
    full_box, gt_volume = _write_fake_bigbird_object(object_root, num_views=5)
    out_dir = tmp_path / "prepared" / "bigbird_advil"

    result = prepare_bigbird_scan(
        object_root=object_root,
        out_dir=out_dir,
        num_views=5,
        config_path=object_root / "bigbird_config.yaml",
        min_valid_depth_pixels=100,
        gt_source="mesh_then_merged_pointcloud",
        gt_voxel_size=0.006,
        repair_mesh=False,
    )
    assert result == out_dir.resolve()
    assert validate_prepared_scan(out_dir) == []

    scan = load_prepared_scan(out_dir)
    assert len(scan.frames) == 5
    assert scan.metadata["dataset"] == "bigbird"
    assert scan.metadata["exact_gt"] is False
    assert scan.gt_volume["gt_type"] == "full_reconstruction_pseudo_gt"
    assert scan.gt_volume["watertight"] is False
    assert "exact_gt" in scan.gt_volume and scan.gt_volume["exact_gt"] is False

    poses = [f.T_cam_to_object for f in scan.frames]
    selected_ids = [int(v) for v in scan.metadata["selected_view_ids"]]
    assert _min_selected_angle_deg(poses, list(range(len(poses)))) > 15.0
    assert len(set(selected_ids)) == 5

    rel_err = abs(scan.gt_volume["volume_m3"] - gt_volume) / gt_volume
    assert rel_err < 0.5


def test_discover_view_candidates(tmp_path: Path):
    object_root = tmp_path / "object"
    _write_fake_bigbird_object(object_root, num_views=6)
    config = BigBirdConfig.from_yaml(object_root / "bigbird_config.yaml")
    candidates = discover_view_candidates(object_root, config)
    assert len(candidates) == 6
    assert all(c.depth_path is not None for c in candidates)

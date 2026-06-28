"""End-to-end method tests on a synthetic prepared scan."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

from tless_volume_benchmark.geometry import make_T
from tless_volume_benchmark.methods.convex_hull import estimate_convex_hull
from tless_volume_benchmark.methods.voxel_carving import estimate_voxel_carving


def _look_at(eye: np.ndarray) -> np.ndarray:
    target = np.zeros(3)
    up = np.array([0.0, 1.0, 0.0])
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = eye
    return np.linalg.inv(T)


def _rasterize_cube(size: float, K: np.ndarray, T_cam_to_object: np.ndarray, shape):
    mesh = trimesh.creation.box(extents=(size, size, size))
    pts, _ = trimesh.sample.sample_surface(mesh, 3000)
    h, w = shape
    T_o2c = np.linalg.inv(T_cam_to_object)
    ones = np.ones((pts.shape[0], 1))
    cam = (T_o2c @ np.hstack([pts, ones]).T).T[:, :3]
    z = cam[:, 2]
    valid = z > 0.01
    u = (K[0, 0] * cam[valid, 0] / z[valid] + K[0, 2]).astype(int)
    v = (K[1, 1] * cam[valid, 1] / z[valid] + K[1, 2]).astype(int)
    depth = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)
    for ui, vi, zi in zip(u, v, z[valid]):
        if 0 <= ui < w and 0 <= vi < h:
            if depth[vi, ui] == 0 or zi < depth[vi, ui]:
                depth[vi, ui] = zi
                mask[vi, ui] = True
    return depth, mask


def _write_synthetic_scan(out: Path, num_views: int = 5) -> float:
    size = 0.12
    gt_m3 = size ** 3
    out.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=(size, size, size))
    mesh.export(out / "gt_mesh.ply")
    gt = {
        "object_id": 1,
        "volume_m3": gt_m3,
        "volume_cm3": gt_m3 * 1e6,
        "gt_type": "mesh_watertight",
        "watertight": True,
        "exact_gt": True,
    }
    with (out / "gt_volume.json").open("w") as f:
        json.dump(gt, f)

    h, w = 128, 128
    K = np.array([[200, 0, w / 2], [0, 200, h / 2], [0, 0, 1]], dtype=float)
    frames = out / "frames"
    frames.mkdir()
    radius = 0.45
    for i in range(num_views):
        ang = 2 * np.pi * i / num_views
        eye = np.array([radius * np.cos(ang), 0.05, radius * np.sin(ang)])
        T = _look_at(eye)
        depth, mask = _rasterize_cube(size, K, T, (h, w))
        prefix = f"frame_{i:03d}"
        cv2.imwrite(str(frames / f"{prefix}_rgb.png"), np.zeros((h, w, 3), dtype=np.uint8))
        np.save(frames / f"{prefix}_depth.npy", depth)
        cv2.imwrite(str(frames / f"{prefix}_mask.png"), mask.astype(np.uint8) * 255)
        np.save(frames / f"{prefix}_K.npy", K)
        np.save(frames / f"{prefix}_T_cam_to_object.npy", T)
        with (frames / f"{prefix}_meta.json").open("w") as f:
            json.dump({"frame": i}, f)
    return gt_m3


def test_convex_hull_on_synthetic_cube(tmp_path: Path):
    pytest.importorskip("scipy", reason="convex hull clustering uses scipy")
    scan_dir = tmp_path / "scan"
    gt_m3 = _write_synthetic_scan(scan_dir, num_views=5)
    report = estimate_convex_hull(scan_dir, voxel_downsample=0.005)
    rel = abs(report["volume_m3"] - gt_m3) / gt_m3
    assert rel < 0.35


def test_voxel_carving_on_synthetic_cube(tmp_path: Path):
    scan_dir = tmp_path / "scan"
    gt_m3 = _write_synthetic_scan(scan_dir, num_views=5)
    report = estimate_voxel_carving(
        scan_dir, voxel_size=0.008, depth_tolerance=0.02, min_views_checked=1
    )
    assert report["volume_m3"] > 0
    rel = abs(report["volume_m3"] - gt_m3) / gt_m3
    assert rel < 0.85


@pytest.mark.skipif(
    __import__("os").environ.get("TLESS_SKIP_TSDF") == "1",
    reason="TSDF disabled",
)
def test_tsdf_on_synthetic_cube(tmp_path: Path):
    from tless_volume_benchmark.methods.tsdf import estimate_tsdf

    scan_dir = tmp_path / "scan"
    gt_m3 = _write_synthetic_scan(scan_dir, num_views=5)
    report = estimate_tsdf(scan_dir, voxel_length=0.004, sdf_trunc=0.02, verbose=False)
    assert report.get("tsdf_backend") == "numpy"
    assert (scan_dir / "outputs" / "tsdf" / "tsdf_mesh_cleaned.ply").is_file()
    if report["volume_m3"] is not None:
        rel = abs(report["volume_m3"] - gt_m3) / gt_m3
        assert rel < 0.55

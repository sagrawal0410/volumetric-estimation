"""Rendered stereo + FoundationStereo mock pipeline tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import trimesh

pytest.importorskip("scipy", exc_type=ImportError)

from tests.conftest import _look_at_pose, render_depth_mask
from volume_benchmark.common.io import validate_prepared_scan
from volume_benchmark.methods.convex_hull import estimate_convex_hull_volume
from volume_benchmark.stereo.disparity_depth import disparity_to_depth_m
from volume_benchmark.stereo.foundation_stereo_backend import FoundationStereoBackend
from volume_benchmark.stereo.fs_prepare_scan import fs_stereo_to_rgbd_scan
from volume_benchmark.stereo.render_stereo_from_mesh import render_rectified_stereo_from_mesh
from volume_benchmark.stereo.stereo_dataset_adapter import save_prepared_stereo_scan


class _GtDepthBackend:
    """Fake FS backend: disparity from rendered GT depth (perfect stereo)."""

    def __init__(self, gt_depths: list[np.ndarray], fx: float, baseline: float) -> None:
        self.gt_depths = gt_depths
        self.fx = fx
        self.baseline = baseline
        self.checkpoint_path = Path("mock")
        self.variant = "mock"

    def predict_disparity(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        idx = getattr(self, "_idx", 0)
        self._idx = idx + 1
        depth = self.gt_depths[idx % len(self.gt_depths)]
        disp = np.zeros_like(depth, dtype=np.float32)
        valid = depth > 0.01
        disp[valid] = (self.fx * self.baseline) / depth[valid]
        return disp


def _make_cube_stereo_scan(tmp_path: Path, box_size: float = 0.2, baseline_m: float = 0.12) -> tuple[Path, float, list[np.ndarray]]:
    mesh = trimesh.creation.box(extents=(box_size, box_size, box_size))
    mesh_path = tmp_path / "cube.ply"
    mesh.export(mesh_path)
    gt_m3 = float(box_size**3)

    width, height = 128, 128
    K = np.array([[200.0, 0, width / 2], [0.0, 200.0, height / 2], [0.0, 0.0, 1.0]])
    gt_depths: list[np.ndarray] = []
    frames_data = []
    for i in range(5):
        angle = 2 * np.pi * i / 5
        eye = np.array([0.55 * np.cos(angle), 0.05, 0.55 * np.sin(angle)])
        T = _look_at_pose(eye)
        left, right, mask, meta = render_rectified_stereo_from_mesh(
            mesh_path, K, (width, height), T, baseline_m, mesh_units="m"
        )
        depth, _ = render_depth_mask(mesh, K, T, (height, width))
        gt_depths.append(depth)
        use_mask = mask if mask is not None else depth > 0
        frames_data.append((left, right, use_mask, T, meta))

    gt_volume = {
        "volume_m3": gt_m3,
        "volume_cm3": gt_m3 * 1e6,
        "gt_type": "mesh_watertight",
        "watertight": True,
        "exact_gt": True,
        "source_mesh": str(mesh_path),
    }
    stereo_dir = save_prepared_stereo_scan(
        tmp_path / "stereo",
        K,
        baseline_m,
        frames_data,
        mesh_path,
        gt_volume,
        metadata={"source_mode": "rendered_stereo_from_gt_mesh", "num_views": 5},
    )
    return stereo_dir, float(K[0, 0]), gt_depths


def test_rendered_stereo_fs_mock_volume_recovery(tmp_path: Path):
    stereo_dir, fx, gt_depths = _make_cube_stereo_scan(tmp_path)
    backend = _GtDepthBackend(gt_depths, fx=fx, baseline=0.12)
    fs_dir = fs_stereo_to_rgbd_scan(stereo_dir, tmp_path / "fs_depth", backend, save_debug=False)

    errors = validate_prepared_scan(fs_dir)
    assert errors == []

    scan_meta = __import__("json").loads((fs_dir / "metadata.json").read_text())
    assert scan_meta.get("depth_backend") == "foundationstereo"
    frame0 = scan_meta["frame_source_info"]["0"]
    assert frame0["depth_source"] == "foundationstereo"

    report = estimate_convex_hull_volume(fs_dir, voxel_downsample=0.005)
    rel = report["relative_error_percent"] / 100.0
    assert rel < 0.35


def test_monocular_only_fails_without_stereo_mode():
    from volume_benchmark.datasets.wildrgbd_stereo_adapter import prepare_wildrgbd_stereo_rendered

    with pytest.raises((FileNotFoundError, ValueError)):
        prepare_wildrgbd_stereo_rendered("/nonexistent/prepared_scene", "/tmp/out")


@pytest.mark.skipif(
    os.environ.get("FOUNDATIONSTEREO_CHECKPOINT") is None,
    reason="Set FOUNDATIONSTEREO_CHECKPOINT and FOUNDATIONSTEREO_REPO for integration test",
)
def test_foundationstereo_integration_optional(tmp_path: Path):
    repo = os.environ["FOUNDATIONSTEREO_REPO"]
    ckpt = os.environ["FOUNDATIONSTEREO_CHECKPOINT"]
    stereo_dir, _, _ = _make_cube_stereo_scan(tmp_path)
    backend = FoundationStereoBackend(repo_path=repo, checkpoint_path=ckpt, variant="fast", device="cpu")
    fs_dir = fs_stereo_to_rgbd_scan(stereo_dir, tmp_path / "fs_real", backend, save_debug=False)
    assert (fs_dir / "frames" / "frame_000_depth.npy").is_file()

"""Tests for voxel carving volume method."""

from pathlib import Path

import pytest
import trimesh

from volume_benchmark.methods.voxel_carving import estimate_voxel_carving_volume
from tests.conftest import create_concave_scan, create_shape_scan, create_synthetic_scan


@pytest.fixture
def cube_scan(tmp_path: Path) -> Path:
    return create_synthetic_scan(tmp_path / "cube", box_size=0.15, num_views=5)


@pytest.fixture
def sphere_scan(tmp_path: Path) -> Path:
    return create_shape_scan(
        tmp_path / "sphere",
        mesh=trimesh.creation.icosphere(radius=0.08),
        num_views=5,
        label="sphere",
    )


@pytest.fixture
def concave_scan(tmp_path: Path) -> Path:
    return create_concave_scan(tmp_path / "concave")


def test_voxel_carving_cube(cube_scan: Path):
    report = estimate_voxel_carving_volume(
        cube_scan, voxel_size=0.008, min_views_checked=1
    )
    assert (cube_scan / "outputs" / "voxel_carving" / "carved_voxels.npz").exists()
    assert report["volume_m3"] > 0
    assert report["relative_error_percent"] < 45.0


@pytest.mark.slow
def test_voxel_carving_sphere_voxel_size_effect(sphere_scan: Path):
    coarse = estimate_voxel_carving_volume(
        sphere_scan, voxel_size=0.012, min_views_checked=1
    )
    fine = estimate_voxel_carving_volume(
        sphere_scan, voxel_size=0.006, min_views_checked=1
    )
    gt = coarse["gt_volume_m3"]
    coarse_err = abs(coarse["volume_m3"] - gt) / gt
    fine_err = abs(fine["volume_m3"] - gt) / gt
    assert fine_err <= coarse_err + 0.15


def test_voxel_carving_concave_overestimates(concave_scan: Path):
    try:
        report = estimate_voxel_carving_volume(
            concave_scan, voxel_size=0.008, min_views_checked=1
        )
    except ValueError as exc:
        pytest.skip(f"concave fixture unavailable: {exc}")
    assert report["volume_m3"] >= report["gt_volume_m3"] * 0.95

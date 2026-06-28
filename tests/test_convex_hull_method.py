"""Tests for convex hull volume method."""

from pathlib import Path

import pytest

pytest.importorskip("scipy", exc_type=ImportError)

from volume_benchmark.methods.convex_hull import estimate_convex_hull_volume
from tests.conftest import create_bop_like_scan, create_synthetic_scan


@pytest.fixture
def cube_scan(tmp_path: Path) -> Path:
    return create_synthetic_scan(tmp_path / "cube", box_size=0.15, num_views=5)


@pytest.fixture
def bop_scan(tmp_path: Path) -> Path:
    return create_bop_like_scan(tmp_path / "bop")


def test_convex_hull_cube_scan(cube_scan: Path):
    report = estimate_convex_hull_volume(cube_scan, voxel_downsample=0.003)
    assert (cube_scan / "outputs" / "convex_hull" / "report.json").exists()
    assert (cube_scan / "outputs" / "convex_hull" / "hull_mesh.ply").exists()
    assert report["volume_m3"] > 0
    assert report["relative_error_percent"] < 35.0
    assert report["expected_bias"] == "usually overestimates non-convex objects"


def test_convex_hull_bop_like_scan(bop_scan: Path):
    report = estimate_convex_hull_volume(bop_scan, voxel_downsample=0.002)
    assert report["num_points"] > 100
    assert report["relative_error_percent"] < 50.0

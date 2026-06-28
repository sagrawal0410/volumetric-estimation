"""Tests for TSDF volume method."""

from pathlib import Path

import pytest
import trimesh

from volume_benchmark.methods.tsdf import estimate_tsdf_volume
from tests.conftest import create_bop_like_scan, create_shape_scan, create_synthetic_scan


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
def cylinder_scan(tmp_path: Path) -> Path:
    return create_shape_scan(
        tmp_path / "cylinder",
        mesh=trimesh.creation.cylinder(radius=0.06, height=0.14),
        num_views=5,
        label="cylinder",
    )


@pytest.fixture
def bop_scan(tmp_path: Path) -> Path:
    return create_bop_like_scan(tmp_path / "bop")


@pytest.mark.slow
def test_tsdf_cube(cube_scan: Path):
    report = estimate_tsdf_volume(cube_scan, voxel_length=0.006, sdf_trunc=0.02)
    assert (cube_scan / "outputs" / "tsdf" / "report.json").exists()
    if report["volume_m3"] is not None:
        assert report["relative_error_percent"] < 60.0


@pytest.mark.slow
def test_tsdf_sphere_and_cylinder(sphere_scan: Path, cylinder_scan: Path):
    for scan in (sphere_scan, cylinder_scan):
        report = estimate_tsdf_volume(scan, voxel_length=0.006)
        assert report["watertight"] or report["volume_m3"] is None


@pytest.mark.slow
def test_tsdf_bop_mini(bop_scan: Path):
    report = estimate_tsdf_volume(bop_scan, voxel_length=0.005)
    assert "tsdf_mesh_cleaned.ply" in report["outputs"]["tsdf_mesh_cleaned"]


def test_tsdf_non_watertight_returns_none_without_repair(tmp_path: Path):
    open_box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    open_mesh = trimesh.Trimesh(vertices=open_box.vertices, faces=open_box.faces[:-2])
    scan = create_shape_scan(tmp_path / "open", mesh=open_mesh, num_views=3, label="open")

    pytest.importorskip("open3d")
    import os

    if os.environ.get("VOLUME_BENCHMARK_SKIP_OPEN3D") == "1":
        pytest.skip("Open3D disabled")

    report = estimate_tsdf_volume(scan, voxel_length=0.008, repair_mesh=False)
    assert report["volume_m3"] is None
    assert report["watertight"] is False
    assert "warning" in report

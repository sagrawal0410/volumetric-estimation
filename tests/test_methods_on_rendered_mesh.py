"""End-to-end method tests on synthetically rendered mesh scans."""

from pathlib import Path

import pytest

pytest.importorskip("scipy", exc_type=ImportError)

from volume_benchmark.common.io import load_prepared_scan, validate_prepared_scan
from volume_benchmark.methods.convex_hull import estimate_convex_hull_volume
from volume_benchmark.methods.voxel_carving import estimate_voxel_carving_volume
from tests.conftest import create_synthetic_scan


@pytest.fixture
def synthetic_scan(tmp_path: Path) -> Path:
    return create_synthetic_scan(tmp_path / "box_scan", box_size=0.2, num_views=5)


def test_prepared_scan_validates(synthetic_scan: Path):
    errors = validate_prepared_scan(synthetic_scan)
    assert errors == []


def test_convex_hull_on_rendered_box(synthetic_scan: Path):
    report = estimate_convex_hull_volume(synthetic_scan, voxel_downsample=0.005)
    rel_err = report["relative_error_percent"] / 100.0
    assert rel_err < 0.35


def test_voxel_carving_on_rendered_box(synthetic_scan: Path):
    report = estimate_voxel_carving_volume(
        synthetic_scan, voxel_size=0.01, padding=0.03, min_views_checked=1
    )
    rel_err = report["relative_error_percent"] / 100.0
    assert rel_err < 0.45


@pytest.mark.slow
def test_tsdf_on_rendered_box(synthetic_scan: Path):
    from volume_benchmark.methods.tsdf import estimate_tsdf_volume

    report = estimate_tsdf_volume(synthetic_scan, voxel_length=0.008)
    if report["volume_m3"] is None:
        pytest.skip("TSDF mesh not watertight in this environment")
    rel_err = report["relative_error_percent"] / 100.0
    assert rel_err < 0.50

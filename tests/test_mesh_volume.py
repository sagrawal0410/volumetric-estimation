"""Tests for mesh volume computation."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from volume_benchmark.common.mesh_volume import (
    compute_convex_hull_volume_m3,
    compute_mesh_volume_m3,
    compute_voxelized_mesh_volume_m3,
    load_mesh_as_meters,
    write_gt_volume_json,
)


def test_box_mesh_volume():
    box = trimesh.creation.box(extents=(0.2, 0.3, 0.4))
    vol, watertight, gt_type = compute_mesh_volume_m3(box)
    expected = 0.2 * 0.3 * 0.4
    assert watertight
    assert gt_type == "mesh_watertight"
    assert abs(vol - expected) < 1e-6


def test_non_watertight_raises_without_repair():
    # Open box: remove one face
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    faces = box.faces[:-2]
    open_mesh = trimesh.Trimesh(vertices=box.vertices, faces=faces)
    assert not open_mesh.is_watertight
    with pytest.raises(ValueError, match="not watertight"):
        compute_mesh_volume_m3(open_mesh, repair=False)


def test_load_mesh_mm_units(tmp_path: Path):
    box = trimesh.creation.box(extents=(100.0, 100.0, 100.0))  # mm
    path = tmp_path / "box_mm.ply"
    box.export(path)
    mesh = load_mesh_as_meters(path, source_units="mm")
    vol, _, _ = compute_mesh_volume_m3(mesh)
    assert abs(vol - 0.001) < 1e-5


def test_voxelized_volume_approximates_box():
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    vol_exact, _, _ = compute_mesh_volume_m3(box)
    vol_vox = compute_voxelized_mesh_volume_m3(box, voxel_size=0.005)
    assert abs(vol_vox - vol_exact) / vol_exact < 0.15


def test_convex_hull_of_box():
    box = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    hull_vol = compute_convex_hull_volume_m3(box)
    exact, _, _ = compute_mesh_volume_m3(box)
    assert abs(hull_vol - exact) < 1e-6


def test_write_gt_volume_json(tmp_path: Path):
    out = tmp_path / "gt_volume.json"
    write_gt_volume_json(out, 0.001, "mesh_watertight", True, "mesh.ply")
    import json

    data = json.loads(out.read_text())
    assert data["volume_cm3"] == pytest.approx(1000.0)
    assert data["gt_type"] == "mesh_watertight"

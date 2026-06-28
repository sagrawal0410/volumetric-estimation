"""Tests for T-LESS mesh volume computation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from tless_volume_benchmark.mesh_volume import compute_mesh_volume_m3, discover_tless_models_dir


def test_discover_models_cad(tmp_path: Path):
    cad = tmp_path / "models_cad"
    cad.mkdir()
    (cad / "models_info.json").write_text("{}", encoding="utf-8")
    (cad / "obj_000001.ply").write_text("ply", encoding="utf-8")
    found = discover_tless_models_dir(tmp_path, preference="cad")
    assert found.name == "models_cad"


def test_cube_mm_to_meters_volume(tmp_path: Path):
    # 100mm cube -> 0.1m cube -> 0.001 m³
    box_mm = trimesh.creation.box(extents=(100.0, 100.0, 100.0))
    mesh_path = tmp_path / "models" / "obj_000001.ply"
    mesh_path.parent.mkdir(parents=True)
    box_mm.export(mesh_path)

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) / 1000.0
    info = compute_mesh_volume_m3(mesh, repair=False)
    assert info["watertight"]
    assert info["volume_m3"] is not None
    assert abs(info["volume_m3"] - 0.001) < 1e-4

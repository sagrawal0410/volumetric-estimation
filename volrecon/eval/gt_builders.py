"""Ground-truth mesh / depth builders for evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from volrecon.config import PROJECT_ROOT
from volrecon.geometry.render_gt import load_and_transform_bop_model
from volrecon.geometry.mesh_volume import voxelize_mesh
from volrecon.io.json_io import read_json


def resolve_path(path: Path | str, root: Path = PROJECT_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root / p).resolve()


def load_gt_scene_mesh(scene_id: str, dataset: str, root: Path = PROJECT_ROOT) -> trimesh.Trimesh | None:
    meta_path = root / "data" / "processed" / dataset / scene_id / "scene_meta.json"
    if not meta_path.exists():
        return None
    meta = read_json(meta_path)
    gt_path = meta.get("scene_gt_mesh_path")
    if not gt_path:
        return None
    mesh = trimesh.load(resolve_path(gt_path, root), force="mesh", process=False)
    return mesh if isinstance(mesh, trimesh.Trimesh) else None


def build_bop_union_gt_mesh(scene_id: str, root: Path = PROJECT_ROOT) -> trimesh.Trimesh | None:
    gt_dir = root / "data" / "processed" / "bop_tless" / scene_id / "gt" / "object_meshes_in_scene_frame"
    if not gt_dir.exists():
        return None
    meshes = []
    for p in sorted(gt_dir.glob("*.ply")):
        m = trimesh.load(p, force="mesh", process=False)
        if isinstance(m, trimesh.Trimesh):
            meshes.append(m)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def load_bop_union_voxels(scene_id: str, root: Path = PROJECT_ROOT) -> dict | None:
    path = root / "data" / "processed" / "bop_tless" / scene_id / "gt" / "union_gt_voxels.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {"voxels": data["voxels"], "voxel_size_m": float(data["voxel_size_m"])}


def gt_volume_from_union_voxels(scene_id: str, root: Path = PROJECT_ROOT) -> float | None:
    data = load_bop_union_voxels(scene_id, root)
    if data is None:
        return None
    return float(data["voxels"].sum()) * (data["voxel_size_m"] ** 3)

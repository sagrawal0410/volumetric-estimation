"""Mesh loading and ground-truth volume for T-LESS object models."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh

GtType = Literal[
    "mesh_watertight",
    "mesh_repaired",
    "mesh_convex_hull_fallback_not_exact",
]


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Basic mesh cleanup."""
    cleaned = mesh.copy()
    cleaned.remove_infinite_values()
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.update_faces(cleaned.unique_faces())
    cleaned.remove_unreferenced_vertices()
    cleaned.merge_vertices()
    return cleaned


GtType = Literal[
    "mesh_watertight",
    "mesh_repaired",
    "mesh_convex_hull_fallback_not_exact",
]

# BOP T-LESS ships multiple model folders (same obj_*.ply names, different meshes).
TLESS_MODEL_DIR_CANDIDATES = (
    "models_cad",      # default for T-LESS in BOP — manual CAD models
    "models",          # legacy / generic BOP layout
    "models_reconst",  # reconstructed from training RGB-D (colored)
    "models_eval",     # decimated/resampled for pose-error metrics — avoid for volume GT
)


def discover_tless_models_dir(
    dataset_root: str | Path,
    preference: str = "cad",
) -> Path:
    """
    Find the T-LESS object models folder under a BOP dataset root.

    Modern BOP archives provide ``models_cad`` and ``models_eval`` (not plain
    ``models/``). Both contain ``obj_000001.ply`` … ``obj_000030.ply`` with the
    same IDs but different mesh geometry.

    For volume ground truth, prefer ``models_cad`` (original CAD). ``models_eval``
    is uniformly decimated for BOP pose-error computation and can bias volume.
    """
    root = Path(dataset_root).expanduser().resolve()
    if preference == "cad":
        order = ("models_cad", "models", "models_reconst", "models_eval")
    elif preference == "eval":
        order = ("models_eval", "models_cad", "models", "models_reconst")
    elif preference == "reconst":
        order = ("models_reconst", "models_cad", "models", "models_eval")
    else:
        order = TLESS_MODEL_DIR_CANDIDATES

    for name in order:
        candidate = root / name
        info = candidate / "models_info.json"
        if candidate.is_dir() and info.is_file():
            return candidate

    raise FileNotFoundError(
        f"No T-LESS models folder under {root}. Looked for: {order}. "
        "Extract tless_models.zip into the dataset root; expect models_cad/ "
        "and models_eval/ (BOP format), not necessarily models/."
    )


def load_tless_model_mesh_meters(
    dataset_root: str | Path,
    object_id: int,
    model_dir: str | None = None,
    model_preference: str = "cad",
    out_gt_mesh_path: Path | None = None,
) -> trimesh.Trimesh:
    """
    Load T-LESS object model and convert vertices from mm to meters.

    If ``model_dir`` is None, auto-discovers ``models_cad`` (preferred) or other
    BOP model folders.

    Saves cleaned meter-scale mesh to out_gt_mesh_path if provided.
    """
    root = Path(dataset_root).expanduser().resolve()
    if model_dir is None:
        models_path = discover_tless_models_dir(root, preference=model_preference)
    else:
        models_path = root / model_dir
    mesh_path = models_path / f"obj_{object_id:06d}.ply"
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"T-LESS model not found: {mesh_path}. "
            f"Available model roots: {[p.name for p in root.iterdir() if p.is_dir() and p.name.startswith('models')]}"
        )

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh from {mesh_path}, got {type(mesh)}")

    mesh = mesh.copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) / 1000.0
    mesh = clean_mesh(mesh)

    if out_gt_mesh_path is not None:
        out_gt_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out_gt_mesh_path)

    return mesh


def compute_mesh_volume_m3(
    mesh: trimesh.Trimesh,
    repair: bool = False,
) -> dict[str, Any]:
    """
    Compute mesh volume in m³.

    Returns dict with volume_m3, volume_cm3, watertight, repaired, bbox_extents_m,
    num_vertices, num_faces, gt_type. volume_m3 is None if not watertight and repair=False.
    """
    mesh = clean_mesh(mesh)
    watertight = bool(mesh.is_watertight)
    repaired = False
    gt_type: GtType = "mesh_watertight"

    if watertight and mesh.volume > 0:
        volume_m3 = abs(float(mesh.volume))
    elif repair:
        try:
            trimesh.repair.fill_holes(mesh)
            mesh.fix_normals()
            mesh.merge_vertices()
            repaired = True
            watertight = bool(mesh.is_watertight)
            if watertight and mesh.volume > 0:
                volume_m3 = abs(float(mesh.volume))
                gt_type = "mesh_repaired"
            else:
                volume_m3 = None
                gt_type = "mesh_repaired"
                warnings.warn("Mesh repair did not yield watertight volume")
        except Exception as exc:
            volume_m3 = None
            warnings.warn(f"Mesh repair failed: {exc}")
            gt_type = "mesh_repaired"
    else:
        volume_m3 = None
        warnings.warn(
            "Mesh is not watertight; refusing to treat volume as exact GT. "
            "Use repair=True or compute_fallback_convex_hull_gt()."
        )
        gt_type = "mesh_watertight"

    bbox = mesh.bounds
    extents = (bbox[1] - bbox[0]).astype(np.float64)

    return {
        "volume_m3": volume_m3,
        "volume_cm3": volume_m3 * 1e6 if volume_m3 is not None else None,
        "watertight": watertight,
        "repaired": repaired,
        "bbox_extents_m": extents.tolist(),
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces)),
        "gt_type": gt_type,
    }


def compute_fallback_convex_hull_gt(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Convex hull fallback — not exact GT for non-convex objects."""
    hull = mesh.convex_hull
    volume_m3 = abs(float(hull.volume))
    return {
        "volume_m3": volume_m3,
        "volume_cm3": volume_m3 * 1e6,
        "watertight": True,
        "repaired": False,
        "gt_type": "mesh_convex_hull_fallback_not_exact",
        "exact_gt": False,
        "num_vertices": int(len(hull.vertices)),
        "num_faces": int(len(hull.faces)),
        "bbox_extents_m": (hull.bounds[1] - hull.bounds[0]).tolist(),
    }


def write_gt_volume_json(
    path: str | Path,
    *,
    object_id: int,
    volume_m3: float,
    gt_type: str,
    watertight: bool,
    exact_gt: bool,
    source_mesh: str,
    split: str | None = None,
    repaired: bool = False,
    extra: dict | None = None,
) -> None:
    """Write gt_volume.json for a prepared scan."""
    payload: dict[str, Any] = {
        "object_id": object_id,
        "volume_m3": volume_m3,
        "volume_cm3": volume_m3 * 1e6,
        "gt_type": gt_type,
        "watertight": watertight,
        "exact_gt": exact_gt,
        "repaired": repaired,
        "source_mesh": source_mesh,
    }
    if split is not None:
        payload["split"] = split
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

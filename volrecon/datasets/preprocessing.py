"""Shared preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from volrecon.config import PreprocessConfig
from volrecon.datasets.canonical_schema import SceneRecord, ViewRecord
from volrecon.io.json_io import write_json, write_jsonl


def write_scene_outputs(
    scene: SceneRecord,
    cfg: PreprocessConfig,
    manifest_path: Path,
    append_manifest: bool = True,
) -> None:
    scene_dir = cfg.processed_root / scene.dataset / scene.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    write_json(scene_dir / "scene_meta.json", scene.to_dict(cfg.project_root))

    rows = [v.to_dict(cfg.project_root) for v in scene.views]
    if append_manifest and manifest_path.exists():
        existing = manifest_path.read_text(encoding="utf-8").strip()
        mode = "a" if existing else "w"
        with manifest_path.open(mode, encoding="utf-8") as f:
            import json

            from volrecon.io.json_io import NumpyEncoder

            for row in rows:
                f.write(json.dumps(row, cls=NumpyEncoder) + "\n")
    else:
        write_jsonl(manifest_path, rows)


def save_placeholder_estimated_depth(view_dir: Path) -> Path:
    """Create an empty placeholder; inference must populate this later."""
    path = view_dir / "estimated_depth_placeholder.npy"
    if not path.exists():
        np.save(path, np.zeros((1, 1), dtype=np.float32))
    return path


def build_view_directory(scene_root: Path, view_id: str) -> Path:
    view_dir = scene_root / "views" / view_id
    view_dir.mkdir(parents=True, exist_ok=True)
    return view_dir


def foundation_stereo_usable(view: ViewRecord) -> bool:
    if view.synthetic and view.stereo and view.stereo.synthetic:
        return bool(view.left_path and view.right_path and view.stereo.baseline_m)
    if view.stereo and view.stereo.has_true_stereo:
        return bool(view.left_path and view.right_path and view.K is not None)
    return False

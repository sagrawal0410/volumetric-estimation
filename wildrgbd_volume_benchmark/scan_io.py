"""Load normalized WildRGB-D prepared scenes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class PreparedFrame:
    index: int
    rgb: np.ndarray
    depth_m: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    T_cam_to_world: np.ndarray
    T_cam_to_object: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedScene:
    scene_dir: Path
    category: str
    scene_id: str
    scene_type: str
    pseudo_gt: dict[str, Any]
    T_world_to_object: np.ndarray
    frames: list[PreparedFrame]
    selected_views: dict[str, Any] | None = None


def _sampled_dir(scene_dir: Path) -> Path:
    p = scene_dir / "sampled_5view"
    if not p.is_dir():
        raise FileNotFoundError(f"Missing sampled_5view/ under {scene_dir}")
    return p


def load_prepared_scene(prepared_scene_dir: str | Path) -> PreparedScene:
    root = Path(prepared_scene_dir).expanduser().resolve()
    meta_path = root / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata.json under {root}")
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    pg_path = root / "pseudo_gt" / "pseudo_gt_volume.json"
    if not pg_path.is_file():
        raise FileNotFoundError(f"Missing pseudo_gt/pseudo_gt_volume.json under {root}")
    with pg_path.open("r", encoding="utf-8") as f:
        pseudo_gt = json.load(f)

    T_world_to_object = np.array(pseudo_gt.get("T_world_to_object", meta.get("T_world_to_object")), dtype=np.float64)
    if T_world_to_object.shape != (4, 4):
        T_world_to_object = np.eye(4)

    sampled = _sampled_dir(root)
    frames_dir = sampled / "frames"
    selected_views = None
    sv_path = sampled / "selected_views.json"
    if sv_path.is_file():
        with sv_path.open("r", encoding="utf-8") as f:
            selected_views = json.load(f)

    indices = sorted(int(p.name.split("_")[1]) for p in frames_dir.glob("frame_*_depth.npy"))
    frames: list[PreparedFrame] = []
    for idx in indices:
        prefix = f"frame_{idx:03d}"
        depth_m = np.load(frames_dir / f"{prefix}_depth.npy")
        mask = cv2.imread(str(frames_dir / f"{prefix}_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
        K = np.load(frames_dir / f"{prefix}_K.npy")
        T_w = np.load(frames_dir / f"{prefix}_T_cam_to_world.npy")
        T_o = np.load(frames_dir / f"{prefix}_T_cam_to_object.npy")
        rgb = cv2.cvtColor(cv2.imread(str(frames_dir / f"{prefix}_rgb.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        fmeta: dict[str, Any] = {}
        mp = frames_dir / f"{prefix}_meta.json"
        if mp.is_file():
            with mp.open("r", encoding="utf-8") as f:
                fmeta = json.load(f)
        frames.append(
            PreparedFrame(
                index=idx,
                rgb=rgb,
                depth_m=depth_m.astype(np.float32),
                mask=mask,
                K=K.astype(np.float64),
                T_cam_to_world=T_w.astype(np.float64),
                T_cam_to_object=T_o.astype(np.float64),
                meta=fmeta,
            )
        )

    if not frames:
        raise ValueError(f"No frames in {frames_dir}")

    return PreparedScene(
        scene_dir=root,
        category=meta.get("category", "unknown"),
        scene_id=meta.get("scene_id", root.name),
        scene_type=meta.get("scene_type", "unknown"),
        pseudo_gt=pseudo_gt,
        T_world_to_object=T_world_to_object,
        frames=frames,
        selected_views=selected_views,
    )


def pseudo_gt_comparison_fields(scene: PreparedScene, volume_m3: float | None) -> dict[str, Any]:
    gt_m3 = scene.pseudo_gt.get("volume_m3")
    gt_cm3 = scene.pseudo_gt.get("volume_cm3")
    if gt_m3 is None or volume_m3 is None:
        return {
            "pseudo_gt_volume_m3": gt_m3,
            "pseudo_gt_volume_cm3": gt_cm3,
            "relative_error_percent": None,
        }
    rel = abs(volume_m3 - gt_m3) / gt_m3 * 100.0 if gt_m3 > 0 else None
    return {
        "pseudo_gt_volume_m3": gt_m3,
        "pseudo_gt_volume_cm3": gt_cm3,
        "relative_error_percent": rel,
    }


def resolve_output_dir(scene_dir: Path, method: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()
    return scene_dir / "outputs" / method


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

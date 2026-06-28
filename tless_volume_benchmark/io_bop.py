"""BOP / T-LESS dataset I/O."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import numpy as np

from tless_volume_benchmark.geometry import (
    bop_pose_m2c_to_T_cam_to_object,
    camera_center_object,
)

SPLIT_ALIASES = {
    "train_primesense": ["train_primesense"],
    "test_primesense": ["test_primesense", "test_primesense_all", "test_primesense_bop19"],
}


@dataclass
class CandidateFrame:
    """One T-LESS observation candidate for view selection."""

    object_id: int
    split: str
    scene_id: str
    image_id: int
    gt_id: int
    rgb: np.ndarray
    depth_m: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    T_cam_to_object: np.ndarray
    visib_fract: float | None
    valid_object_depth_pixels: int
    camera_center_object: np.ndarray = field(repr=False)
    depth_scale: float = 1.0
    rgb_path: str = ""
    depth_path: str = ""
    mask_path: str = ""


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_bop_depth_m(depth_path: str | Path, depth_scale: float) -> np.ndarray:
    """
    Load BOP uint16 depth PNG and convert to meters.

    depth_m = raw.astype(float32) * depth_scale / 1000.0
    Invalid raw == 0 becomes 0.0.
    """
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    depth_m = raw.astype(np.float32) * float(depth_scale) / 1000.0
    depth_m[raw == 0] = 0.0
    return depth_m


def read_mask(mask_path: str | Path) -> np.ndarray:
    """Load mask PNG as boolean array."""
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")
    return raw > 0


def read_rgb(rgb_or_gray_path: str | Path) -> np.ndarray:
    """Load RGB image (BGR from OpenCV, converted to RGB)."""
    img = cv2.imread(str(rgb_or_gray_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read RGB image: {rgb_or_gray_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_scene_camera(scene_dir: Path) -> dict[int, dict[str, Any]]:
    """
    Load scene_camera.json.

    Returns dict image_id -> {K, depth_scale, optional cam_R_w2c, cam_t_w2c}.
    """
    path = scene_dir / "scene_camera.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing scene_camera.json in {scene_dir}")
    raw = load_json(path)
    out: dict[int, dict[str, Any]] = {}
    for key, entry in raw.items():
        im_id = int(key)
        K = np.array(entry["cam_K"], dtype=np.float64).reshape(3, 3)
        depth_scale = float(entry.get("depth_scale", 1.0))
        info: dict[str, Any] = {"K": K, "depth_scale": depth_scale}
        if "cam_R_w2c" in entry:
            info["cam_R_w2c"] = np.array(entry["cam_R_w2c"], dtype=np.float64).reshape(3, 3)
        if "cam_t_w2c" in entry:
            info["cam_t_w2c"] = np.array(entry["cam_t_w2c"], dtype=np.float64).reshape(3)
        out[im_id] = info
    return out


def load_scene_gt(scene_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """Load scene_gt.json: image_id -> list of GT annotations."""
    path = scene_dir / "scene_gt.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing scene_gt.json in {scene_dir}. "
            "Use train_primesense or a T-LESS archive with public GT."
        )
    raw = load_json(path)
    return {int(k): v for k, v in raw.items()}


def load_scene_gt_info(scene_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """Load scene_gt_info.json: image_id -> list of per-instance metadata."""
    path = scene_dir / "scene_gt_info.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing scene_gt_info.json in {scene_dir}. "
            "Use train_primesense or a T-LESS archive with public GT."
        )
    raw = load_json(path)
    return {int(k): v for k, v in raw.items()}


def find_mask_path(
    scene_dir: Path,
    image_id: int,
    gt_id: int,
    prefer_visible: bool = True,
) -> Path | None:
    """Find mask path, preferring mask_visib over mask."""
    im = f"{image_id:06d}"
    gt = f"{gt_id:06d}"
    visib = scene_dir / "mask_visib" / f"{im}_{gt}.png"
    full = scene_dir / "mask" / f"{im}_{gt}.png"
    if prefer_visible and visib.is_file():
        return visib
    if full.is_file():
        return full
    if visib.is_file():
        return visib
    return None


def _resolve_split_dir(dataset_root: Path, split: str) -> Path:
    aliases = SPLIT_ALIASES.get(split, [split])
    for name in aliases:
        candidate = dataset_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Split {split!r} not found under {dataset_root}. "
        f"Tried: {aliases}. Download and extract the appropriate T-LESS archive."
    )


def _count_valid_object_depth(depth_m: np.ndarray, mask: np.ndarray) -> int:
    valid = mask & (depth_m > 0.01) & np.isfinite(depth_m)
    return int(np.count_nonzero(valid))


def _find_rgb_path(scene_dir: Path, image_id: int) -> Path | None:
    im = f"{image_id:06d}"
    rgb_dir = scene_dir / "rgb"
    for ext in (".png", ".jpg", ".jpeg"):
        path = rgb_dir / f"{im}{ext}"
        if path.is_file():
            return path
    return None


def _find_depth_path(scene_dir: Path, image_id: int) -> Path | None:
    im = f"{image_id:06d}"
    depth_dir = scene_dir / "depth"
    for ext in (".png", ".tif", ".tiff"):
        path = depth_dir / f"{im}{ext}"
        if path.is_file():
            return path
    return None


def _scene_matches_object(scene_dir: Path, object_id: int) -> bool:
    """T-LESS train scenes are named by object id; test scenes are numeric."""
    name = scene_dir.name
    if name.isdigit():
        return True
    m = re.match(r"^obj[_-]?(\d+)$", name, re.IGNORECASE)
    if m:
        return int(m.group(1)) == object_id
    m = re.match(r"^(\d+)$", name)
    if m and int(m.group(1)) == object_id:
        return True
    return name == f"{object_id:06d}"


def iter_tless_candidates(
    dataset_root: str | Path,
    split: str,
    object_id: int,
    min_visib_fract: float = 0.0,
    prefer_visible_mask: bool = True,
    min_valid_depth_pixels: int = 100,
) -> Iterator[CandidateFrame]:
    """
    Iterate candidate frames for one T-LESS object across scenes in a split.

    Yields CandidateFrame for each image/GT annotation matching object_id.
    """
    root = Path(dataset_root).expanduser().resolve()
    split_dir = _resolve_split_dir(root, split)

    if split.startswith("test") and not (split_dir / "scene_gt.json").exists():
        sample_scene = next((p for p in split_dir.iterdir() if p.is_dir()), None)
        if sample_scene and not (sample_scene / "scene_gt.json").exists():
            raise FileNotFoundError(
                f"No scene_gt.json found under {split_dir}. "
                "Test splits without GT cannot be used for volume benchmark preparation. "
                "Use train_primesense or download test_primesense_bop19 with GT."
            )

    scene_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if not scene_dirs:
        raise FileNotFoundError(f"No scene folders under {split_dir}")

    for scene_dir in scene_dirs:
        if split.startswith("train") and not _scene_matches_object(scene_dir, object_id):
            continue

        try:
            scene_camera = load_scene_camera(scene_dir)
            scene_gt = load_scene_gt(scene_dir)
            scene_gt_info = load_scene_gt_info(scene_dir)
        except FileNotFoundError as exc:
            if split.startswith("test"):
                continue
            raise

        scene_id = scene_dir.name
        for image_id, annotations in scene_gt.items():
            gt_infos = scene_gt_info.get(image_id, [])
            if image_id not in scene_camera:
                continue

            cam = scene_camera[image_id]
            K = cam["K"]
            depth_scale = cam["depth_scale"]

            depth_path = _find_depth_path(scene_dir, image_id)
            rgb_path = _find_rgb_path(scene_dir, image_id)
            if depth_path is None:
                continue

            depth_m = read_bop_depth_m(depth_path, depth_scale)
            rgb = read_rgb(rgb_path) if rgb_path else np.zeros((*depth_m.shape, 3), dtype=np.uint8)

            for ann_idx, ann in enumerate(annotations):
                if int(ann["obj_id"]) != object_id:
                    continue
                gt_id = int(ann.get("gt_id", ann_idx))

                mask_path = find_mask_path(scene_dir, image_id, gt_id, prefer_visible_mask)
                if mask_path is None:
                    if prefer_visible_mask:
                        visib_dir = scene_dir / "mask_visib"
                        if not visib_dir.is_dir():
                            raise FileNotFoundError(
                                f"Missing mask_visib/ in {scene_dir}. "
                                "Download a T-LESS archive with visible masks or set prefer_visible_mask=False."
                            )
                    continue

                mask = read_mask(mask_path)
                visib_fract: float | None = None
                if ann_idx < len(gt_infos):
                    info = gt_infos[ann_idx]
                    if "visib_fract" in info:
                        visib_fract = float(info["visib_fract"])
                        if visib_fract < min_visib_fract:
                            continue

                valid_pixels = _count_valid_object_depth(depth_m, mask)
                if valid_pixels < min_valid_depth_pixels:
                    continue

                T_cam_to_object = bop_pose_m2c_to_T_cam_to_object(
                    ann["cam_R_m2c"], ann["cam_t_m2c"]
                )
                cam_center = camera_center_object(T_cam_to_object)

                yield CandidateFrame(
                    object_id=object_id,
                    split=split,
                    scene_id=scene_id,
                    image_id=image_id,
                    gt_id=gt_id,
                    rgb=rgb,
                    depth_m=depth_m,
                    mask=mask,
                    K=K,
                    T_cam_to_object=T_cam_to_object,
                    visib_fract=visib_fract,
                    valid_object_depth_pixels=valid_pixels,
                    camera_center_object=cam_center,
                    depth_scale=depth_scale,
                    rgb_path=str(rgb_path) if rgb_path else "",
                    depth_path=str(depth_path),
                    mask_path=str(mask_path),
                )

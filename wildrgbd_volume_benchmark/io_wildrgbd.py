"""WildRGB-D dataset I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from wildrgbd_volume_benchmark.masks import load_mask
from wildrgbd_volume_benchmark.metadata import load_metadata


@dataclass
class WildRGBDFrame:
    frame_id: str
    rgb_path: str
    depth_path: str
    mask_path: str
    rgb: np.ndarray | None = None
    depth_m: np.ndarray | None = None
    mask: np.ndarray | None = None
    K: np.ndarray | None = None
    T_cam_to_world: np.ndarray | None = None


@dataclass
class WildRGBDScene:
    category: str
    scene_id: str
    scene_dir: str
    scene_type: str
    frames: list[WildRGBDFrame] = field(default_factory=list)
    K: np.ndarray | None = None
    image_size: tuple[int, int] | None = None


def load_depth_m(depth_path: str | Path) -> np.ndarray:
    """Load uint16 depth PNG; depth_m = raw / 1000.0; zero -> 0."""
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read depth: {depth_path}")
    depth_m = raw.astype(np.float32) / 1000.0
    depth_m[raw == 0] = 0.0
    return depth_m


def load_rgb(rgb_path: str | Path) -> np.ndarray:
    img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read RGB: {rgb_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_cam_poses(scene_dir: str | Path) -> dict[str, np.ndarray]:
    """Parse cam_poses.txt: frame_id followed by 16 floats (row-major 4x4 T_cam_to_world)."""
    path = Path(scene_dir) / "cam_poses.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cam_poses.txt in {scene_dir}")

    poses: dict[str, np.ndarray] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 17:
            raise ValueError(f"{path}:{line_no}: expected frame_id + 16 floats, got {len(parts)} parts")
        frame_id = parts[0]
        vals = np.array([float(x) for x in parts[1:17]], dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"{path}:{line_no}: non-finite pose values")
        T = vals.reshape(4, 4)
        if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-4):
            raise ValueError(f"{path}:{line_no}: last row not [0,0,0,1]: {T[3]}")
        poses[frame_id] = T
    if not poses:
        raise ValueError(f"No poses parsed from {path}")
    return poses


def load_types_json(category_dir: str | Path) -> dict[str, str]:
    path = Path(category_dir) / "types.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"types.json must be a dict, got {type(raw)}")
    return {str(k): str(v) for k, v in raw.items()}


def _scene_folder_name(scene_id: str) -> str:
    return scene_id if scene_id.startswith("scenes_") else f"scenes_{scene_id}"


def _list_frame_ids(rgb_dir: Path) -> list[str]:
    ids = sorted(p.stem for p in rgb_dir.glob("*.png"))
    if not ids:
        ids = sorted(p.stem for p in rgb_dir.glob("*.jpg"))
    return ids


def load_scene(
    scene_dir: str | Path,
    category: str,
    scene_id: str,
    scene_type: str = "unknown",
    load_images: bool = False,
) -> WildRGBDScene:
    scene_dir = Path(scene_dir).resolve()
    K, image_size = load_metadata(scene_dir)
    poses = load_cam_poses(scene_dir)

    rgb_dir = scene_dir / "rgb"
    depth_dir = scene_dir / "depth"
    mask_dir = scene_dir / "masks"

    frames: list[WildRGBDFrame] = []
    for fid in _list_frame_ids(rgb_dir):
        if fid not in poses:
            continue
        rgb_path = rgb_dir / f"{fid}.png"
        if not rgb_path.is_file():
            rgb_path = rgb_dir / f"{fid}.jpg"
        depth_path = depth_dir / f"{fid}.png"
        mask_path = mask_dir / f"{fid}.png"

        if not depth_path.is_file() or not mask_path.is_file():
            continue

        frame = WildRGBDFrame(
            frame_id=fid,
            rgb_path=str(rgb_path),
            depth_path=str(depth_path),
            mask_path=str(mask_path),
            K=K.copy(),
            T_cam_to_world=poses[fid].copy(),
        )
        if load_images:
            frame.rgb = load_rgb(rgb_path)
            frame.depth_m = load_depth_m(depth_path)
            frame.mask = load_mask(mask_path)
        frames.append(frame)

    if not frames:
        raise ValueError(f"No matched rgb/depth/mask/pose frames in {scene_dir}")

    return WildRGBDScene(
        category=category,
        scene_id=scene_id,
        scene_dir=str(scene_dir),
        scene_type=scene_type,
        frames=frames,
        K=K,
        image_size=image_size,
    )


def discover_scenes(
    root_dir: str | Path,
    categories: Sequence[str] | None = None,
    scene_types: Sequence[str] = ("single",),
) -> list[WildRGBDScene]:
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"WildRGB-D root not found: {root}")

    cat_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if categories:
        allowed = {c.lower() for c in categories}
        cat_dirs = [p for p in cat_dirs if p.name.lower() in allowed]

    scenes: list[WildRGBDScene] = []
    for cat_dir in cat_dirs:
        types_map = load_types_json(cat_dir)
        scenes_root = cat_dir / "scenes"
        if not scenes_root.is_dir():
            continue
        for scene_path in sorted(scenes_root.iterdir()):
            if not scene_path.is_dir() or not scene_path.name.startswith("scenes_"):
                continue
            scene_id = scene_path.name
            stype = types_map.get(scene_id.replace("scenes_", ""), types_map.get(scene_id, "unknown"))
            if scene_types and stype not in scene_types:
                continue
            try:
                scenes.append(
                    load_scene(scene_path, category=cat_dir.name, scene_id=scene_id, scene_type=stype)
                )
            except (FileNotFoundError, ValueError):
                continue
    return scenes


def iter_scene_frames(
    scene: WildRGBDScene,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[WildRGBDFrame]:
    selected = scene.frames[:: max(1, frame_stride)]
    if max_frames is not None:
        selected = selected[:max_frames]
    for frame in selected:
        if frame.depth_m is None:
            frame.depth_m = load_depth_m(frame.depth_path)
        if frame.mask is None:
            frame.mask = load_mask(frame.mask_path)
        if frame.rgb is None and frame.rgb_path:
            frame.rgb = load_rgb(frame.rgb_path)
        if frame.K is None:
            frame.K = scene.K.copy() if scene.K is not None else load_metadata(Path(scene.scene_dir))[0]
        yield frame

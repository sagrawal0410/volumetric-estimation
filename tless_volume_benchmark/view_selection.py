"""View selection for T-LESS multi-view volume estimation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from tless_volume_benchmark.io_bop import CandidateFrame


def angular_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two 3D direction vectors."""
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b) + 1e-12)
    cos = float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _view_direction(cam_center: np.ndarray) -> np.ndarray:
    """Direction from object origin toward camera center."""
    d = cam_center - np.zeros(3)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return d / n


def _pairwise_angles_deg(selected: list[CandidateFrame]) -> list[dict]:
    dirs = [_view_direction(c.camera_center_object) for c in selected]
    pairs = []
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            pairs.append(
                {
                    "i": i,
                    "j": j,
                    "angle_deg": angular_distance_deg(dirs[i], dirs[j]),
                }
            )
    return pairs


def select_views(
    candidates: Sequence[CandidateFrame],
    num_views: int = 5,
    min_angle_deg: float = 20.0,
    prefer_clean: bool = True,
) -> list[CandidateFrame]:
    """
    Select diverse high-quality views of the same object.

    Greedy score after seeding best coverage:
      0.50 * min angular distance to selected
      0.30 * normalized valid object depth pixels
      0.20 * visib_fract (if available)
    """
    if not candidates:
        return []

    pool = list(candidates)
    if prefer_clean:
        pool = [c for c in pool if c.valid_object_depth_pixels > 0]
    if not pool:
        return []

    max_pixels = max(c.valid_object_depth_pixels for c in pool)
    selected: list[CandidateFrame] = []
    used_keys: set[tuple[str, int]] = set()

    def key(c: CandidateFrame) -> tuple[str, int]:
        return (c.scene_id, c.image_id)

    # Seed: largest valid depth coverage
    seed = max(pool, key=lambda c: c.valid_object_depth_pixels)
    selected.append(seed)
    used_keys.add(key(seed))
    remaining = [c for c in pool if key(c) not in used_keys]

    angle_thresh = min_angle_deg
    while len(selected) < num_views and remaining:
        best: CandidateFrame | None = None
        best_score = -1.0

        sel_dirs = [_view_direction(c.camera_center_object) for c in selected]

        for cand in remaining:
            if key(cand) in used_keys:
                continue
            cand_dir = _view_direction(cand.camera_center_object)
            min_angle = min(angular_distance_deg(cand_dir, d) for d in sel_dirs)
            if min_angle < angle_thresh:
                continue

            norm_pixels = cand.valid_object_depth_pixels / max(max_pixels, 1)
            visib = cand.visib_fract if cand.visib_fract is not None else 0.0
            score = 0.50 * (min_angle / 180.0) + 0.30 * norm_pixels + 0.20 * visib
            if score > best_score:
                best_score = score
                best = cand

        if best is not None:
            selected.append(best)
            used_keys.add(key(best))
            remaining = [c for c in remaining if key(c) not in used_keys]
        else:
            if angle_thresh <= 5.0:
                break
            angle_thresh = max(5.0, angle_thresh - 5.0)

    # Fill remaining slots without angle constraint if needed
    while len(selected) < num_views and remaining:
        best = max(
            remaining,
            key=lambda c: (
                c.valid_object_depth_pixels,
                c.visib_fract if c.visib_fract is not None else 0.0,
            ),
        )
        if key(best) in used_keys:
            break
        selected.append(best)
        used_keys.add(key(best))
        remaining = [c for c in remaining if key(c) not in used_keys]

    return selected


def build_selected_views_json(selected: list[CandidateFrame]) -> dict:
    """Build selected_views.json payload."""
    entries = []
    for idx, c in enumerate(selected):
        entries.append(
            {
                "frame_index": idx,
                "object_id": c.object_id,
                "split": c.split,
                "scene_id": c.scene_id,
                "image_id": c.image_id,
                "gt_id": c.gt_id,
                "visib_fract": c.visib_fract,
                "valid_object_depth_pixels": c.valid_object_depth_pixels,
                "camera_center_object": c.camera_center_object.tolist(),
                "depth_path": c.depth_path,
                "mask_path": c.mask_path,
                "rgb_path": c.rgb_path,
            }
        )
    return {
        "num_views": len(selected),
        "views": entries,
        "pairwise_angles_deg": _pairwise_angles_deg(selected),
    }


def save_selected_views_json(path: str | Path, selected: list[CandidateFrame]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(build_selected_views_json(selected), f, indent=2)

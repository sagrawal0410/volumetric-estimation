"""Sparse view selection for WildRGB-D scenes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from wildrgbd_volume_benchmark.geometry import camera_center_from_T
from wildrgbd_volume_benchmark.io_wildrgbd import WildRGBDFrame, WildRGBDScene, load_depth_m, load_mask


@dataclass
class FrameQuality:
    frame_id: str
    valid_mask_pixels: int
    valid_depth_pixels: int
    mask_area: int
    depth_median: float
    pose_valid: bool
    score: float


def angular_distance_between_camera_centers(c1: np.ndarray, c2: np.ndarray, origin: np.ndarray | None = None) -> float:
    origin = np.zeros(3) if origin is None else origin
    d1 = c1 - origin
    d2 = c2 - origin
    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(d1 / n1, d2 / n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def compute_frame_quality(frame: WildRGBDFrame) -> FrameQuality:
    if frame.depth_m is None:
        frame.depth_m = load_depth_m(frame.depth_path)
    if frame.mask is None:
        frame.mask = load_mask(frame.mask_path)

    mask = frame.mask
    depth = frame.depth_m
    valid_mask = int(mask.sum())
    valid_depth = int((mask & (depth > 0.01) & np.isfinite(depth)).sum())
    depth_vals = depth[mask & (depth > 0.01)]
    depth_med = float(np.median(depth_vals)) if depth_vals.size else 0.0
    pose_valid = frame.T_cam_to_world is not None and np.all(np.isfinite(frame.T_cam_to_world))

    score = (
        0.4 * (valid_depth / max(valid_mask, 1))
        + 0.3 * min(valid_depth / 5000.0, 1.0)
        + 0.2 * min(valid_mask / 10000.0, 1.0)
        + 0.1 * (1.0 if pose_valid else 0.0)
    )
    return FrameQuality(
        frame_id=frame.frame_id,
        valid_mask_pixels=valid_mask,
        valid_depth_pixels=valid_depth,
        mask_area=valid_mask,
        depth_median=depth_med,
        pose_valid=pose_valid,
        score=score,
    )


def _frame_index_map(scene: WildRGBDScene) -> dict[str, int]:
    return {f.frame_id: i for i, f in enumerate(scene.frames)}


def select_sparse_views(
    scene: WildRGBDScene,
    T_world_to_object: np.ndarray,
    num_views: int = 5,
    min_angle_deg: float = 20.0,
    max_frame_gap_preference: bool = True,
    require_valid_depth_pixels: int = 1000,
    prefer_scene_type: str = "single",
) -> list[WildRGBDFrame]:
    if prefer_scene_type and scene.scene_type not in (prefer_scene_type, "unknown"):
        pass  # caller may filter; still allow selection

    qualities = {f.frame_id: compute_frame_quality(f) for f in scene.frames}
    pool = [
        f for f in scene.frames
        if qualities[f.frame_id].valid_depth_pixels >= require_valid_depth_pixels
        and qualities[f.frame_id].pose_valid
    ]
    if not pool:
        pool = [f for f in scene.frames if qualities[f.frame_id].pose_valid]
    if not pool:
        raise ValueError(f"No valid frames for view selection in {scene.scene_id}")

    idx_map = _frame_index_map(scene)
    max_depth = max(qualities[f.frame_id].valid_depth_pixels for f in pool)

    def cam_center_obj(frame: WildRGBDFrame) -> np.ndarray:
        assert frame.T_cam_to_world is not None
        T_cam_to_object = T_world_to_object @ frame.T_cam_to_world
        return camera_center_from_T(T_cam_to_object)

    seed = max(pool, key=lambda f: qualities[f.frame_id].score)
    selected: list[WildRGBDFrame] = [seed]
    used_ids = {seed.frame_id}
    angle_thresh = min_angle_deg

    while len(selected) < num_views:
        best: WildRGBDFrame | None = None
        best_score = -1.0
        sel_centers = [cam_center_obj(f) for f in selected]

        for cand in pool:
            if cand.frame_id in used_ids:
                continue
            c_center = cam_center_obj(cand)
            min_angle = min(
                angular_distance_between_camera_centers(c_center, sc) for sc in sel_centers
            )
            if min_angle < angle_thresh:
                continue

            q = qualities[cand.frame_id]
            norm_depth = q.valid_depth_pixels / max(max_depth, 1)
            temporal = 0.0
            if max_frame_gap_preference and len(selected) >= 1:
                gaps = [abs(idx_map[cand.frame_id] - idx_map[s.frame_id]) for s in selected]
                temporal = min(max(max(gaps) / max(len(scene.frames), 1), 0.0), 1.0)
            mask_stab = min(q.mask_area / 10000.0, 1.0)

            score = (
                0.55 * (min_angle / 180.0)
                + 0.25 * norm_depth
                + 0.10 * temporal
                + 0.10 * mask_stab
            )
            if score > best_score:
                best_score = score
                best = cand

        if best is not None:
            selected.append(best)
            used_ids.add(best.frame_id)
        else:
            if angle_thresh <= 5.0:
                break
            angle_thresh = max(5.0, angle_thresh - 5.0)

    while len(selected) < num_views:
        remaining = [f for f in pool if f.frame_id not in used_ids]
        if not remaining:
            break
        best = max(remaining, key=lambda f: qualities[f.frame_id].score)
        selected.append(best)
        used_ids.add(best.frame_id)

    return selected


def build_selected_views_json(
    scene: WildRGBDScene,
    selected: Sequence[WildRGBDFrame],
    T_world_to_object: np.ndarray,
) -> dict:
    entries = []
    centers = []
    for f in selected:
        q = compute_frame_quality(f)
        assert f.T_cam_to_world is not None
        T_cam_to_object = T_world_to_object @ f.T_cam_to_world
        center = camera_center_from_T(T_cam_to_object)
        centers.append(center)
        entries.append(
            {
                "frame_id": f.frame_id,
                "quality_score": q.score,
                "valid_depth_pixels": q.valid_depth_pixels,
                "valid_mask_pixels": q.valid_mask_pixels,
                "camera_center_object": center.tolist(),
                "T_cam_to_world": f.T_cam_to_world.tolist(),
                "T_cam_to_object": T_cam_to_object.tolist(),
            }
        )

    pairs = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            pairs.append(
                {
                    "i": i,
                    "j": j,
                    "angle_deg": angular_distance_between_camera_centers(centers[i], centers[j]),
                }
            )

    return {
        "category": scene.category,
        "scene_id": scene.scene_id,
        "scene_type": scene.scene_type,
        "num_views": len(selected),
        "views": entries,
        "pairwise_angles_deg": pairs,
    }


def save_selected_views_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

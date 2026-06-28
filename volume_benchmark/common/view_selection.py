"""Select diverse RGB-D views for volume estimation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from volume_benchmark.common.geometry import invert_T


@dataclass
class CandidateFrame:
    """One RGB-D view candidate for multi-view volume estimation."""

    depth_m: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    T_cam_to_object: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.depth_m = np.asarray(self.depth_m, dtype=np.float32)
        self.mask = np.asarray(self.mask, dtype=bool)
        self.K = np.asarray(self.K, dtype=np.float64)
        self.T_cam_to_object = np.asarray(self.T_cam_to_object, dtype=np.float64)
        if self.T_cam_to_object.shape != (4, 4):
            raise ValueError(
                f"T_cam_to_object must be (4, 4), got {self.T_cam_to_object.shape}"
            )

    @property
    def valid_object_depth_pixels(self) -> int:
        if "valid_object_depth_pixels" in self.metadata:
            return int(self.metadata["valid_object_depth_pixels"])
        valid = self.mask & np.isfinite(self.depth_m) & (self.depth_m > 0.01)
        return int(valid.sum())

    @property
    def mask_pixel_count(self) -> int:
        return int(self.mask.sum())

    @property
    def visible_fraction(self) -> Optional[float]:
        value = self.metadata.get("visible_fraction")
        return None if value is None else float(value)

    @property
    def frame_id(self) -> str:
        return str(self.metadata.get("frame_id", ""))

    @property
    def scene_id(self) -> str:
        return str(self.metadata.get("scene_id", ""))


def camera_center_object(T_cam_to_object: np.ndarray) -> np.ndarray:
    """Camera origin expressed in the object coordinate frame (meters)."""
    T_cam_to_object = np.asarray(T_cam_to_object, dtype=np.float64)
    T_object_to_cam = invert_T(T_cam_to_object)
    return T_object_to_cam[:3, 3]


def view_direction_object(T_cam_to_object: np.ndarray) -> np.ndarray:
    """
    Unit vector from the object origin toward the camera.

    Used as a proxy for viewing direction when ranking angular spread.
    """
    center = camera_center_object(T_cam_to_object)
    norm = float(np.linalg.norm(center))
    if norm < 1e-9:
        raise ValueError("Camera center is at the object origin; direction is undefined")
    return center / norm


def angular_distance_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 3-D vectors."""
    v1 = np.asarray(v1, dtype=np.float64).reshape(3)
    v2 = np.asarray(v2, dtype=np.float64).reshape(3)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        raise ValueError("Cannot compute angle for zero-length vector")
    cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _min_angle_to_selected_deg(direction: np.ndarray, selected_directions: list[np.ndarray]) -> float:
    if not selected_directions:
        return 180.0
    return min(angular_distance_deg(direction, sel) for sel in selected_directions)


def _elevation_deg(direction: np.ndarray) -> float:
    """Elevation above the object XZ plane (positive Y is up)."""
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    horiz = np.linalg.norm(direction[[0, 2]])
    if horiz < 1e-9:
        return 90.0 if direction[1] > 0 else -90.0
    return float(np.degrees(np.arctan2(direction[1], horiz)))


def _filter_candidates(
    candidates: list[CandidateFrame],
    min_valid_depth_pixels: int,
    min_mask_pixels: int,
    min_visible_fraction: float,
) -> list[CandidateFrame]:
    filtered: list[CandidateFrame] = []
    for cand in candidates:
        T = cand.T_cam_to_object
        if not np.all(np.isfinite(T)):
            continue
        if cand.valid_object_depth_pixels < min_valid_depth_pixels:
            continue
        if cand.mask_pixel_count < min_mask_pixels:
            continue
        vis = cand.visible_fraction
        if vis is not None and vis < min_visible_fraction:
            continue
        filtered.append(cand)
    return filtered


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def select_diverse_views(
    candidates: list[CandidateFrame],
    num_views: int = 5,
    min_valid_depth_pixels: int = 1000,
    min_angle_deg: float = 25.0,
    prefer_high_elevation: bool = True,
    min_mask_pixels: int = 100,
    min_visible_fraction: float = 0.0,
    selected_views_path: str | Path | None = None,
) -> list[CandidateFrame]:
    """
    Select diverse views for volume estimation.

    Filters candidates, seeds with the view having the most valid object depth
    pixels, then greedily adds views balancing angular spread, pixel coverage,
    and optional visibility. Writes ``selected_views.json`` when
    ``selected_views_path`` is provided.
    """
    if num_views <= 0:
        raise ValueError(f"num_views must be positive, got {num_views}")
    if not candidates:
        raise ValueError("At least one candidate frame is required")

    eligible = _filter_candidates(
        candidates,
        min_valid_depth_pixels=min_valid_depth_pixels,
        min_mask_pixels=min_mask_pixels,
        min_visible_fraction=min_visible_fraction,
    )
    if not eligible:
        best = max(candidates, key=lambda c: c.valid_object_depth_pixels)
        raise ValueError(
            f"No candidates passed filtering (min_valid_depth_pixels={min_valid_depth_pixels}, "
            f"min_mask_pixels={min_mask_pixels}). Best view has "
            f"{best.valid_object_depth_pixels} valid depth pixels."
        )

    num_views = min(num_views, len(eligible))
    if num_views == len(eligible):
        selected = list(eligible)
        _maybe_write_selected_views_json(
            selected, min_angle_deg, min_angle_deg, selected_views_path
        )
        return selected

    directions = [view_direction_object(c.T_cam_to_object) for c in eligible]
    pixel_counts = [c.valid_object_depth_pixels for c in eligible]
    pixel_scores = _normalize_scores([float(v) for v in pixel_counts])

    seed_idx = int(np.argmax(pixel_counts))
    selected_indices = [seed_idx]
    selected_directions = [directions[seed_idx]]
    has_top_view = _elevation_deg(selected_directions[0]) >= 15.0

    angle_threshold = float(min_angle_deg)
    while len(selected_indices) < num_views:
        remaining = [i for i in range(len(eligible)) if i not in selected_indices]
        if not remaining:
            break

        best_idx: Optional[int] = None
        best_score = -1.0

        while best_idx is None and angle_threshold >= 0.0:
            for idx in remaining:
                spread_deg = _min_angle_to_selected_deg(directions[idx], selected_directions)
                if spread_deg < angle_threshold:
                    continue

                spread_score = spread_deg / 180.0
                vis = eligible[idx].visible_fraction
                vis_score = vis if vis is not None else 1.0

                elevation_bonus = 0.0
                if prefer_high_elevation and not has_top_view:
                    if _elevation_deg(directions[idx]) >= 15.0:
                        elevation_bonus = 0.1

                total = (
                    0.5 * spread_score
                    + 0.3 * pixel_scores[idx]
                    + 0.2 * vis_score
                    + elevation_bonus
                )
                if total > best_score:
                    best_score = total
                    best_idx = idx

            if best_idx is None:
                angle_threshold -= 5.0

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        selected_directions.append(directions[best_idx])
        if _elevation_deg(directions[best_idx]) >= 15.0:
            has_top_view = True

    selected = [eligible[i] for i in selected_indices]
    _maybe_write_selected_views_json(
        selected, min_angle_deg, max(angle_threshold, 0.0), selected_views_path
    )
    return selected


def _maybe_write_selected_views_json(
    selected: list[CandidateFrame],
    requested_min_angle_deg: float,
    used_min_angle_deg: float,
    path: str | Path | None,
) -> None:
    if path is None:
        return

    directions = [view_direction_object(c.T_cam_to_object) for c in selected]
    centers = [camera_center_object(c.T_cam_to_object) for c in selected]

    view_records: list[dict[str, Any]] = []
    for i, cand in enumerate(selected):
        min_angle = (
            min(
                angular_distance_deg(directions[i], directions[j])
                for j in range(len(selected))
                if j != i
            )
            if len(selected) > 1
            else 180.0
        )
        record: dict[str, Any] = {
            "index": i,
            "frame_id": cand.frame_id,
            "scene_id": cand.scene_id,
            "valid_object_depth_pixels": cand.valid_object_depth_pixels,
            "visible_fraction": cand.visible_fraction,
            "camera_center_object": centers[i].tolist(),
            "view_direction_object": directions[i].tolist(),
            "elevation_deg": _elevation_deg(directions[i]),
            "min_angle_to_other_selected_deg": min_angle,
        }
        record.update({k: v for k, v in cand.metadata.items() if k not in record})
        view_records.append(record)

    pairwise = []
    for i in range(len(selected)):
        row = []
        for j in range(len(selected)):
            if i == j:
                row.append(0.0)
            else:
                row.append(angular_distance_deg(directions[i], directions[j]))
        pairwise.append(row)

    payload = {
        "num_selected": len(selected),
        "requested_min_angle_deg": requested_min_angle_deg,
        "used_min_angle_deg": used_min_angle_deg,
        "views": view_records,
        "pairwise_angles_deg": pairwise,
    }

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def select_spread_views(
    poses_T_cam_to_object: list[np.ndarray],
    num_views: int,
) -> list[int]:
    """
    Greedy index selection by angular spread (legacy API).

    Returns indices into ``poses_T_cam_to_object``.
    """
    if num_views <= 0:
        raise ValueError(f"num_views must be positive, got {num_views}")
    if not poses_T_cam_to_object:
        raise ValueError("At least one pose is required")

    dummy_depth = np.ones((8, 8), dtype=np.float32)
    dummy_mask = np.ones((8, 8), dtype=bool)
    dummy_K = np.eye(3, dtype=np.float64)

    candidates = [
        CandidateFrame(
            depth_m=dummy_depth,
            mask=dummy_mask,
            K=dummy_K,
            T_cam_to_object=T,
            metadata={"valid_object_depth_pixels": 10_000 - i, "frame_id": str(i)},
        )
        for i, T in enumerate(poses_T_cam_to_object)
    ]
    selected = select_diverse_views(
        candidates,
        num_views=num_views,
        min_valid_depth_pixels=0,
        min_mask_pixels=0,
        min_angle_deg=0.0,
        prefer_high_elevation=False,
    )
    id_to_index = {str(i): i for i in range(len(poses_T_cam_to_object))}
    return [id_to_index[c.frame_id] for c in selected]


def select_bigbird_views(
    poses_T_cam_to_object: list[np.ndarray],
    valid_pixel_counts: list[int],
    num_views: int,
    min_valid_depth_pixels: int = 1000,
) -> list[int]:
    """Legacy BigBIRD index selection API."""
    if len(poses_T_cam_to_object) != len(valid_pixel_counts):
        raise ValueError("poses and valid_pixel_counts must have the same length")

    dummy_depth = np.ones((8, 8), dtype=np.float32)
    dummy_mask = np.ones((8, 8), dtype=bool)
    dummy_K = np.eye(3, dtype=np.float64)

    candidates = [
        CandidateFrame(
            depth_m=dummy_depth,
            mask=dummy_mask,
            K=dummy_K,
            T_cam_to_object=T,
            metadata={
                "valid_object_depth_pixels": count,
                "frame_id": str(i),
            },
        )
        for i, (T, count) in enumerate(zip(poses_T_cam_to_object, valid_pixel_counts, strict=True))
    ]
    selected = select_diverse_views(
        candidates,
        num_views=num_views,
        min_valid_depth_pixels=min_valid_depth_pixels,
        min_mask_pixels=0,
        min_angle_deg=20.0,
        prefer_high_elevation=True,
    )
    id_to_index = {str(i): i for i in range(len(poses_T_cam_to_object))}
    return [id_to_index[c.frame_id] for c in selected]


def select_uniform_indices(num_total: int, num_views: int) -> list[int]:
    """Evenly spaced frame indices (fallback when poses are unavailable)."""
    if num_views <= 0:
        raise ValueError(f"num_views must be positive, got {num_views}")
    if num_total <= 0:
        raise ValueError(f"num_total must be positive, got {num_total}")
    num_views = min(num_views, num_total)
    if num_views == num_total:
        return list(range(num_total))
    return list(np.linspace(0, num_total - 1, num=num_views, dtype=int))

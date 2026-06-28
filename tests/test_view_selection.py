"""Tests for multi-view selection heuristics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from volume_benchmark.common.view_selection import (
    CandidateFrame,
    angular_distance_deg,
    camera_center_object,
    select_diverse_views,
    view_direction_object,
)
from tests.conftest import _look_at_pose


def _make_candidate(
    eye: np.ndarray,
    valid_pixels: int,
    frame_id: str,
    visible_fraction: float | None = None,
    image_size: tuple[int, int] = (64, 64),
) -> CandidateFrame:
    height, width = image_size
    depth = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    depth[: valid_pixels // width + 1, :] = 0.5
    mask[: valid_pixels // width + 1, :] = True
    K = np.array([[200.0, 0, width / 2], [0, 200, height / 2], [0, 0, 1]], dtype=np.float64)
    meta: dict = {
        "valid_object_depth_pixels": valid_pixels,
        "frame_id": frame_id,
        "scene_id": "synthetic",
    }
    if visible_fraction is not None:
        meta["visible_fraction"] = visible_fraction
    return CandidateFrame(
        depth_m=depth,
        mask=mask,
        K=K,
        T_cam_to_object=_look_at_pose(eye),
        metadata=meta,
    )


def _circle_candidates(
    num: int,
    radius: float = 0.6,
    base_pixels: int = 5000,
) -> list[CandidateFrame]:
    candidates = []
    for i in range(num):
        angle = 2 * np.pi * i / num
        eye = np.array([radius * np.cos(angle), 0.0, radius * np.sin(angle)])
        candidates.append(
            _make_candidate(eye, base_pixels - i * 50, frame_id=str(i))
        )
    return candidates


def test_camera_center_and_view_direction():
    eye = np.array([0.0, 0.0, 1.0])
    T = _look_at_pose(eye)
    center = camera_center_object(T)
    direction = view_direction_object(T)
    assert np.allclose(np.linalg.norm(direction), 1.0, atol=1e-6)
    assert angular_distance_deg(direction, direction) == pytest.approx(0.0, abs=1e-6)
    assert np.linalg.norm(center) > 0.5


def test_select_diverse_views_spreads_around_circle():
    candidates = _circle_candidates(num=12)
    selected = select_diverse_views(
        candidates,
        num_views=5,
        min_valid_depth_pixels=1000,
        min_angle_deg=25.0,
    )
    assert len(selected) == 5
    assert selected[0].valid_object_depth_pixels == max(c.valid_object_depth_pixels for c in candidates)

    directions = [view_direction_object(c.T_cam_to_object) for c in selected]
    min_pair_angle = 180.0
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            min_pair_angle = min(min_pair_angle, angular_distance_deg(directions[i], directions[j]))
    assert min_pair_angle > 20.0


def test_duplicate_nearby_views_not_selected_when_alternatives_exist():
    candidates = _circle_candidates(num=8, base_pixels=4000)
    # Add two cameras nearly co-located but with high pixel counts.
    near_a = _make_candidate(np.array([0.6, 0.0, 0.01]), 8000, frame_id="near_a")
    near_b = _make_candidate(np.array([0.6, 0.0, -0.01]), 7900, frame_id="near_b")
    candidates.extend([near_a, near_b])

    selected = select_diverse_views(
        candidates,
        num_views=5,
        min_valid_depth_pixels=1000,
        min_angle_deg=25.0,
    )
    ids = {c.frame_id for c in selected}
    assert not ("near_a" in ids and "near_b" in ids)


def test_select_diverse_views_writes_json(tmp_path: Path):
    candidates = _circle_candidates(num=8)
    out = tmp_path / "selected_views.json"
    select_diverse_views(
        candidates,
        num_views=4,
        min_valid_depth_pixels=1000,
        selected_views_path=out,
    )
    data = json.loads(out.read_text())
    assert data["num_selected"] == 4
    assert len(data["views"]) == 4
    assert len(data["pairwise_angles_deg"]) == 4


def test_filter_rejects_low_visible_fraction():
    candidates = _circle_candidates(num=6)
    for i, c in enumerate(candidates):
        c.metadata["visible_fraction"] = 0.1 if i % 2 == 0 else 0.8

    selected = select_diverse_views(
        candidates,
        num_views=3,
        min_valid_depth_pixels=1000,
        min_visible_fraction=0.5,
    )
    assert all((c.visible_fraction or 0.0) >= 0.5 for c in selected)

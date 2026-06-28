"""Tests for manifest schema and modality rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from volrecon.datasets.canonical_schema import StereoCalibration, ViewRecord


def test_gt_depth_eval_only():
    rec = ViewRecord(
        dataset="bop_tless",
        scene_id="000001",
        view_id="000000",
        rgb_path="data/processed/bop_tless/000001/views/000000/rgb.png",
        gt_depth_path="data/processed/bop_tless/000001/views/000000/gt_depth.png",
        K=np.eye(3),
    )
    assert "gt_depth" in rec.eval_only_modalities
    assert "gt_depth" not in rec.inference_allowed_modalities


def test_manifest_serialization_roundtrip():
    rec = ViewRecord(
        dataset="robi",
        scene_id="scene_a",
        view_id="001",
        left_path="data/processed/robi/scene_a/views/001/left.png",
        right_path="data/processed/robi/scene_a/views/001/right.png",
        K=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64),
        stereo=StereoCalibration(has_true_stereo=True, source="test"),
    )
    d = rec.to_dict()
    rec2 = ViewRecord.from_dict(d)
    assert rec2.scene_id == "scene_a"
    assert rec2.stereo is not None
    assert rec2.stereo.has_true_stereo
    assert "gt_depth" not in rec2.inference_allowed_modalities


def test_inference_modalities_exclude_depth_even_if_available():
    rec = ViewRecord(
        dataset="bop_tless",
        scene_id="s",
        view_id="v",
        gt_depth_path="/tmp/gt_depth.png",
    )
    rec._refresh_modality_lists()
    assert "gt_depth" in rec.available_modalities
    assert "gt_depth" not in rec.inference_allowed_modalities

"""Tests that ZED manifests never include ZED SDK depth modalities."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from volrecon.camera.zed_capture import ZEDCaptureConfig, ZEDStereoCapture
from volrecon.camera.zed_manifest import FORBIDDEN_MODALITIES_ZED, frame_to_view_record, build_view_meta
from volrecon.io.json_io import read_jsonl


@pytest.fixture(autouse=True)
def mock_zed(monkeypatch):
    monkeypatch.setenv("VOLRECON_MOCK_ZED", "1")


def test_view_meta_forbids_zed_depth():
    calib = {
        "K_left": np.eye(3),
        "K_right": np.eye(3),
        "T_left_right": np.eye(4),
        "baseline_m": 0.12,
        "image_width": 640,
        "image_height": 480,
    }
    meta = build_view_meta(
        0, 0, {"left": "views/000000/left.png", "right": "views/000000/right.png"},
        calib, None, "ok", "123", "mock",
    )
    assert "zed_depth" in meta["forbidden_modalities"]
    assert "zed_depth" not in meta["inference_allowed_modalities"]
    assert "left_rgb" in meta["inference_allowed_modalities"]


def test_view_record_no_gt_depth(tmp_path):
    calib = {
        "K_left": np.eye(3),
        "K_right": np.eye(3),
        "T_left_right": np.eye(4),
        "baseline_m": 0.12,
        "image_width": 640,
        "image_height": 480,
    }
    meta = build_view_meta(0, 0, {"left": "l.png", "right": "r.png"}, calib, None, "ok", "1", "mock")
    rec = frame_to_view_record("scene", "000000", {"left": "l.png", "right": "r.png"}, calib, meta)
    assert rec.gt_depth_path is None
    assert "gt_depth" not in rec.inference_allowed_modalities
    assert rec.stereo is not None
    assert rec.stereo.has_true_stereo


def test_capture_manifest_rows(tmp_path):
    capture = ZEDStereoCapture(ZEDCaptureConfig(enable_positional_tracking=True))
    capture.open()
    try:
        scene_dir = tmp_path / "test_scene"
        scene_dir.mkdir()
        (scene_dir / "views").mkdir()
        frame = capture.grab_frame()
        assert frame is not None
        paths = capture.save_frame(frame, scene_dir, "000000")
        rec = capture.build_view_record(frame, paths, "test_scene", "000000")
        from volrecon.camera.zed_manifest import write_scene_manifest

        write_scene_manifest(scene_dir, [rec], tmp_path)
        rows = list(read_jsonl(scene_dir / "manifest.jsonl"))
        assert rows[0]["dataset"] == "zed_live"
        assert rows[0]["stereo"]["has_true_stereo"] is True
        assert rows[0].get("gt_depth_path") is None
    finally:
        capture.close()

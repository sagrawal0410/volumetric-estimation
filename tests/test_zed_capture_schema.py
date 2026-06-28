"""Tests for ZED capture schema and frame structure."""

from __future__ import annotations

import os

import numpy as np
import pytest

from volrecon.camera.camera_health import ZEDDepthCallGuard
from volrecon.camera.zed_capture import ZEDCaptureConfig, ZEDStereoCapture, ZEDStereoFrame


@pytest.fixture(autouse=True)
def mock_zed(monkeypatch):
    monkeypatch.setenv("VOLRECON_MOCK_ZED", "1")
    from volrecon.camera import zed_mock

    zed_mock.Camera._frame_counter = 0


def test_stereo_frame_schema():
    cfg = ZEDCaptureConfig(enable_positional_tracking=True)
    cap = ZEDStereoCapture(cfg)
    cap.open()
    try:
        frame = cap.grab_frame()
        assert frame is not None
        assert isinstance(frame, ZEDStereoFrame)
        assert frame.left_rgb.dtype == np.uint8
        assert frame.right_rgb.dtype == np.uint8
        assert frame.left_rgb.shape[2] == 3
        assert frame.K_left.shape == (3, 3)
        assert frame.K_right.shape == (3, 3)
        assert frame.T_left_right.shape == (4, 4)
        assert frame.baseline_m > 0
        assert frame.tracking_state in {"ok", "unreliable", "disabled"}
    finally:
        cap.close()


def test_depth_guard_blocks_retrieve_measure():
    from volrecon.camera.zed_mock import Camera

    guarded = ZEDDepthCallGuard(Camera())
    with pytest.raises(RuntimeError, match="retrieve_measure is forbidden"):
        guarded.retrieve_measure(None, None)

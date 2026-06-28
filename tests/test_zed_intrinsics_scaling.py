"""Tests for ZED intrinsics scaling."""

from __future__ import annotations

import os

import pytest

from volrecon.camera.zed_calibration import extract_zed_calibration
from volrecon.camera.zed_mock import Camera, InitParameters


@pytest.fixture(autouse=True)
def mock_zed(monkeypatch):
    monkeypatch.setenv("VOLRECON_MOCK_ZED", "1")


def test_intrinsics_scale_on_resize():
    cam = Camera()
    cam.open(InitParameters())
    calib_full = extract_zed_calibration(cam, None)
    calib_half = extract_zed_calibration(cam, (640, 360))

    sx = 640 / calib_full["source_width"]
    sy = 360 / calib_full["source_height"]
    assert abs(calib_half["K_left"][0, 0] - calib_full["K_left"][0, 0] * sx) < 1e-6
    assert abs(calib_half["K_left"][1, 1] - calib_full["K_left"][1, 1] * sy) < 1e-6
    assert abs(calib_half["K_left"][0, 2] - calib_full["K_left"][0, 2] * sx) < 1e-6
    assert abs(calib_half["K_left"][1, 2] - calib_full["K_left"][1, 2] * sy) < 1e-6
    assert calib_half["baseline_m"] == calib_full["baseline_m"]
    cam.close()

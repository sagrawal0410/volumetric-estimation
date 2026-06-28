"""Static and runtime checks that ZED depth APIs are never called."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from volrecon.camera.camera_health import (
    EXCLUDED_FROM_SCAN,
    FORBIDDEN_ZED_PATTERNS,
    GUARDED_DIRS,
    ZEDDepthCallGuard,
    assert_no_zed_depth_calls_enabled,
)


def test_no_forbidden_patterns_in_guarded_code():
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for rel in GUARDED_DIRS:
        base = root / rel
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if path.name in EXCLUDED_FROM_SCAN or path.name.startswith("test_"):
                continue
            text = path.read_text(encoding="utf-8")
            for pat in FORBIDDEN_ZED_PATTERNS:
                if pat in text:
                    offenders.append(f"{path.relative_to(root)}: {pat}")
    assert not offenders, "Forbidden ZED depth API strings found:\n" + "\n".join(offenders)


def test_assert_no_zed_depth_calls_enabled_passes():
    assert_no_zed_depth_calls_enabled()


@pytest.fixture
def mock_zed(monkeypatch):
    monkeypatch.setenv("VOLRECON_MOCK_ZED", "1")
    from volrecon.camera import zed_mock

    zed_mock.Camera._frame_counter = 0


def test_capture_does_not_call_retrieve_measure(mock_zed):
    from volrecon.camera.zed_capture import ZEDCaptureConfig, ZEDStereoCapture

    cap = ZEDStereoCapture(ZEDCaptureConfig(enable_positional_tracking=True))
    cap.open()
    try:
        frame = cap.grab_frame()
        assert frame is not None
        # Guard is active on cap.zed
        with pytest.raises(RuntimeError):
            cap.zed.retrieve_measure(None, None)
    finally:
        cap.close()


def test_live_pipeline_import_no_depth():
    from volrecon.deployment.live_pipeline import LiveReconstructionPipeline  # noqa: F401

    assert LiveReconstructionPipeline is not None

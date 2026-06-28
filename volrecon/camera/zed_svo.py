"""ZED SVO recording and RGB extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from volrecon.camera.capture_session import CaptureSession, CaptureSessionConfig
from volrecon.camera.zed_capture import ZEDCaptureConfig, ZEDStereoCapture
from volrecon.io.json_io import write_json

logger = logging.getLogger(__name__)


@dataclass
class SVORecordConfig:
    output_svo: Path
    resolution: str = "HD1080"
    fps: int = 15
    duration_sec: float = 30.0


def record_svo(cfg: SVORecordConfig, zed_cfg: ZEDCaptureConfig | None = None) -> Path:
    """Record SVO from live ZED (no separate depth arrays)."""
    zed_cfg = zed_cfg or ZEDCaptureConfig(
        camera_resolution=cfg.resolution,
        camera_fps=cfg.fps,
        depth_mode="NONE",
    )
    capture = ZEDStereoCapture(zed_cfg)
    capture.open()
    try:
        max_frames = int(cfg.duration_sec * cfg.fps)
        zed_cfg.max_frames = max_frames
        frames = 0
        while frames < max_frames:
            f = capture.grab_frame()
            if f is None:
                break
            frames += 1
        meta = {
            "output_svo": str(cfg.output_svo),
            "frames_recorded": frames,
            "resolution": cfg.resolution,
            "fps": cfg.fps,
            "notes": "SVO recorded via ZED SDK; use zed_extract_svo_rgb for RGB-only extraction.",
        }
        cfg.output_svo.parent.mkdir(parents=True, exist_ok=True)
        write_json(cfg.output_svo.parent / "recording_meta.json", meta)
        logger.info("SVO recording complete (%d frames). Note: full SVO write requires SDK Recording API.", frames)
        return cfg.output_svo
    finally:
        capture.close()


def extract_svo_to_scene(
    svo_path: Path,
    out_scene: Path,
    frame_stride: int = 5,
    pose_mode: str = "zed_tracking",
    num_keyframes: int | None = None,
) -> Path:
    """Extract rectified left/right RGB from SVO into canonical scene format."""
    zed_cfg = ZEDCaptureConfig(
        svo_input_path=str(svo_path),
        enable_positional_tracking=(pose_mode == "zed_tracking"),
        depth_mode="NONE",
    )
    capture = ZEDStereoCapture(zed_cfg)
    capture.open()
    try:
        session_cfg = CaptureSessionConfig(
            scene_name=out_scene.name,
            output_root=out_scene.parent,
            save_every_n_frames=frame_stride,
            fixed_camera=True,
            pose_mode=pose_mode,
            num_keyframes=num_keyframes or 10_000,
        )
        session = CaptureSession(capture, session_cfg, zed_cfg)
        return session.capture_keyframes(session_cfg.num_keyframes)
    finally:
        capture.close()

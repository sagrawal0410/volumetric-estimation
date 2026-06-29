"""Live ZED deployment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from volrecon.camera.zed_capture import ZEDCaptureConfig


@dataclass
class StereoDepthConfig:
    method: str = "foundation_stereo"
    backend: str = "auto"
    foundationstereo_repo: Path = Path("/path/to/FoundationStereo")
    checkpoint: Path = Path("/path/to/model_best_bp2.pth")
    min_depth_m: float = 0.2
    max_depth_m: float = 4.0
    valid_iters: int = 16
    max_disp: int = 192
    scale: float = 0.5


@dataclass
class FusionLiveConfig:
    method: str = "plain_tsdf"
    voxel_length_m: float = 0.003
    sdf_trunc_m: float = 0.015
    depth_trunc_m: float = 4.0
    extract_mesh_every: int = 5
    min_weight_for_mesh: float = 2.0


@dataclass
class LivePipelineConfig:
    mode: str = "capture_then_reconstruct"
    scene_name: str = "zed_scene"
    output_root: Path = Path("data/zed_captures")
    zed: ZEDCaptureConfig = field(default_factory=ZEDCaptureConfig)
    capture_num_keyframes: int = 30
    pose_mode: str = "zed_tracking"
    stereo_depth: StereoDepthConfig = field(default_factory=StereoDepthConfig)
    fusion: FusionLiveConfig = field(default_factory=FusionLiveConfig)
    overwrite: bool = False
    overwrite_depth: bool = False
    overwrite_recon: bool = False
    dry_run: bool = False
    allow_no_pose_single_view: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "LivePipelineConfig":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        cam = data.get("camera", {})
        cap = data.get("capture", {})
        pose = data.get("pose", {})
        sd = data.get("stereo_depth", {})
        fus = data.get("fusion", {})
        zed = ZEDCaptureConfig(
            camera_resolution=cam.get("resolution", "HD1080"),
            camera_fps=cam.get("fps", 15),
            depth_mode=cam.get("depth_mode", "NONE"),
            output_resolution_scale=cam.get("output_resolution_scale", 1.0),
            serial_number=cam.get("serial_number"),
            enable_positional_tracking=pose.get("mode", "zed_tracking") == "zed_tracking",
        )
        return cls(
            scene_name=cap.get("scene_name") or "zed_scene",
            output_root=Path(data.get("output", {}).get("root", "data/zed_captures")),
            zed=zed,
            capture_num_keyframes=cap.get("num_keyframes", 30),
            pose_mode=pose.get("mode", "zed_tracking"),
            stereo_depth=StereoDepthConfig(
                backend=sd.get("backend", "auto"),
                foundationstereo_repo=Path(sd.get("foundationstereo_repo", "/path/to/FoundationStereo")),
                checkpoint=Path(sd.get("checkpoint", "/path/to/model_best_bp2.pth")),
                min_depth_m=sd.get("min_depth_m", 0.2),
                max_depth_m=sd.get("max_depth_m", 4.0),
                valid_iters=sd.get("valid_iters", 16),
                max_disp=sd.get("max_disp", 192),
                scale=sd.get("scale", 0.5),
            ),
            fusion=FusionLiveConfig(
                method=fus.get("method", "weighted_tsdf"),
                voxel_length_m=fus.get("voxel_length_m", 0.003),
                sdf_trunc_m=fus.get("sdf_trunc_m", 0.015),
                depth_trunc_m=fus.get("depth_trunc_m", 4.0),
                extract_mesh_every=fus.get("extract_mesh_every", 5),
            ),
            allow_no_pose_single_view=pose.get("allow_no_pose_single_view", False),
        )

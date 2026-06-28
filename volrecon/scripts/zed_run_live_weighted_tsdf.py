"""Live weighted TSDF from ZED capture."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from volrecon.deployment.live_config import LivePipelineConfig
from volrecon.deployment.live_pipeline import LiveReconstructionPipeline
from volrecon.camera.zed_capture import ZEDCaptureConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="ZED live weighted TSDF pipeline.")
    parser.add_argument("--out", default="data/zed_captures")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--resolution", default="HD1080")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--pose_mode", default="zed_tracking")
    parser.add_argument("--foundationstereo_repo", required=True, type=Path)
    parser.add_argument("--foundationstereo_ckpt", required=True, type=Path)
    parser.add_argument("--num_keyframes", type=int, default=40)
    parser.add_argument("--voxel_length_m", type=float, default=0.003)
    parser.add_argument("--sdf_trunc_m", type=float, default=0.015)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock:
        os.environ["VOLRECON_MOCK_ZED"] = "1"

    cfg = LivePipelineConfig.from_yaml(args.config) if args.config else LivePipelineConfig()
    cfg.scene_name = args.scene_name
    cfg.output_root = Path(args.out)
    cfg.capture_num_keyframes = args.num_keyframes
    cfg.pose_mode = args.pose_mode
    cfg.zed = ZEDCaptureConfig(camera_resolution=args.resolution, camera_fps=args.fps, enable_positional_tracking=True)
    cfg.stereo_depth.foundationstereo_repo = args.foundationstereo_repo
    cfg.stereo_depth.checkpoint = args.foundationstereo_ckpt
    cfg.fusion.method = "weighted_tsdf"
    cfg.fusion.voxel_length_m = args.voxel_length_m
    cfg.fusion.sdf_trunc_m = args.sdf_trunc_m

    LiveReconstructionPipeline(cfg).run_live_incremental_weighted_tsdf()


if __name__ == "__main__":
    main()

"""Capture then run FoundationStereo + TSDF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from volrecon.deployment.live_config import LivePipelineConfig, StereoDepthConfig, FusionLiveConfig
from volrecon.deployment.live_pipeline import LiveReconstructionPipeline
from volrecon.camera.zed_capture import ZEDCaptureConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="ZED capture + reconstruct end-to-end.")
    parser.add_argument("--out", default="data/zed_captures")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--resolution", default="HD1080")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--num_keyframes", type=int, default=30)
    parser.add_argument("--pose_mode", default="zed_tracking")
    parser.add_argument("--foundationstereo_repo", required=True, type=Path)
    parser.add_argument("--foundationstereo_ckpt", required=True, type=Path)
    parser.add_argument("--method", default="plain_tsdf", choices=["plain_tsdf", "weighted_tsdf"])
    parser.add_argument("--voxel_length_m", type=float, default=0.003)
    parser.add_argument("--sdf_trunc_m", type=float, default=0.015)
    parser.add_argument("--depth_min_m", type=float, default=0.2)
    parser.add_argument("--depth_max_m", type=float, default=4.0)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.mock:
        os.environ["VOLRECON_MOCK_ZED"] = "1"

    if args.config and args.config.exists():
        cfg = LivePipelineConfig.from_yaml(args.config)
        cfg.scene_name = args.scene_name
        cfg.output_root = Path(args.out)
    else:
        cfg = LivePipelineConfig(
            scene_name=args.scene_name,
            output_root=Path(args.out),
            zed=ZEDCaptureConfig(camera_resolution=args.resolution, camera_fps=args.fps, enable_positional_tracking=True),
            capture_num_keyframes=args.num_keyframes,
            pose_mode=args.pose_mode,
            stereo_depth=StereoDepthConfig(
                foundationstereo_repo=args.foundationstereo_repo,
                checkpoint=args.foundationstereo_ckpt,
                min_depth_m=args.depth_min_m,
                max_depth_m=args.depth_max_m,
            ),
            fusion=FusionLiveConfig(method=args.method, voxel_length_m=args.voxel_length_m, sdf_trunc_m=args.sdf_trunc_m),
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )

    pipeline = LiveReconstructionPipeline(cfg)
    if args.method == "weighted_tsdf":
        pipeline.run_live_incremental_weighted_tsdf()
    else:
        pipeline.run_capture_then_reconstruct()


if __name__ == "__main__":
    main()

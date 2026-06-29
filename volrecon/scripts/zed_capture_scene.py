"""Capture a ZED scene to canonical format."""

from __future__ import annotations

import argparse
import os

from rich.console import Console

from volrecon.camera.zed_capture import ZEDCaptureConfig
from volrecon.deployment.live_pipeline import LiveReconstructionPipeline
from volrecon.deployment.live_config import LivePipelineConfig

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture ZED stereo RGB scene.")
    parser.add_argument("--out", default="data/zed_captures")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--resolution", default="HD1080")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--num_keyframes", type=int, default=30)
    parser.add_argument("--pose_mode", default="zed_tracking", choices=["zed_tracking", "fixed_rig_yaml", "external_poses", "none"])
    parser.add_argument("--min_translation_between_keyframes_m", type=float, default=0.03)
    parser.add_argument("--min_rotation_between_keyframes_deg", type=float, default=5.0)
    parser.add_argument("--save_preview_video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock:
        os.environ["VOLRECON_MOCK_ZED"] = "1"

    cfg = LivePipelineConfig(
        scene_name=args.scene_name,
        output_root=__import__("pathlib").Path(args.out),
        zed=ZEDCaptureConfig(
            camera_resolution=args.resolution,
            camera_fps=args.fps,
            enable_positional_tracking=(args.pose_mode == "zed_tracking"),
            min_translation_between_keyframes_m=args.min_translation_between_keyframes_m,
            min_rotation_between_keyframes_deg=args.min_rotation_between_keyframes_deg,
            save_preview_video=args.save_preview_video,
        ),
        capture_num_keyframes=args.num_keyframes,
        pose_mode=args.pose_mode,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    pipeline = LiveReconstructionPipeline(cfg)
    scene_dir = pipeline.run_capture_only()
    console.print(f"Scene: {scene_dir}")
    console.print(f"Manifest: {scene_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()

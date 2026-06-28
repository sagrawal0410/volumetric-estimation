"""Run plain TSDF fusion from depth predictions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from volrecon.config import PROJECT_ROOT
from volrecon.fusion.open3d_tsdf import PlainTSDFConfig
from volrecon.fusion.plain_tsdf import PlainTSDFRunConfig, run_plain_tsdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plain Open3D TSDF fusion.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--depth_predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--voxel_length_m", type=float, default=0.003)
    parser.add_argument("--sdf_trunc_m", type=float, default=0.015)
    parser.add_argument("--depth_trunc_m", type=float, default=2.0)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--max_views_per_scene", type=int, default=40)
    parser.add_argument("--use_gt_bounds_for_debug", action="store_true")
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    tsdf_cfg = PlainTSDFConfig(
        voxel_length_m=args.voxel_length_m,
        sdf_trunc_m=args.sdf_trunc_m,
        depth_trunc_m=args.depth_trunc_m,
        frame_stride=args.frame_stride,
        max_views=args.max_views_per_scene,
    )
    if args.config and args.config.exists():
        with args.config.open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        tsdf_cfg = PlainTSDFConfig(
            voxel_length_m=y.get("voxel_length_m", tsdf_cfg.voxel_length_m),
            sdf_trunc_m=y.get("sdf_trunc_m", tsdf_cfg.sdf_trunc_m),
            depth_trunc_m=y.get("depth_trunc_m", tsdf_cfg.depth_trunc_m),
            min_depth_m=y.get("min_depth_m", tsdf_cfg.min_depth_m),
            max_depth_m=y.get("max_depth_m", tsdf_cfg.max_depth_m),
            frame_stride=y.get("frame_stride", tsdf_cfg.frame_stride),
            max_views=y.get("max_views_per_scene", tsdf_cfg.max_views),
            mesh_cleanup=y.get("mesh_cleanup", tsdf_cfg.mesh_cleanup),
        )

    run_cfg = PlainTSDFRunConfig(
        manifest_path=args.manifest,
        depth_predictions_root=args.depth_predictions,
        out_root=args.out,
        tsdf=tsdf_cfg,
        project_root=args.project_root,
        use_gt_bounds_for_debug=args.use_gt_bounds_for_debug,
    )
    outputs = run_plain_tsdf(run_cfg)
    logger.info("Finished TSDF for %d scenes", len(outputs))


if __name__ == "__main__":
    main()

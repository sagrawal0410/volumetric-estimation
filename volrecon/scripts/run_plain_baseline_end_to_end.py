"""End-to-end plain baseline: FoundationStereo -> TSDF -> eval."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from volrecon.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run plain baseline end-to-end.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--foundationstereo_repo", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max_scenes", type=int, default=5)
    parser.add_argument("--max_views_per_scene", type=int, default=20)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--valid_iters", type=int, default=16)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    out = args.out
    depth_out = out / "depth_predictions"
    recon_out = out / "reconstructions"
    eval_out = out / "eval_report"

    min_depth, max_depth = 0.1, 2.0
    if args.config and args.config.exists():
        with args.config.open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        min_depth = y.get("min_depth_m", min_depth)
        max_depth = y.get("max_depth_m", max_depth)

    # Optionally trim manifest to max_scenes
    manifest_use = args.manifest
    if args.max_scenes:
        from volrecon.io.json_io import read_jsonl, write_jsonl

        rows = read_jsonl(args.manifest)
        scenes = sorted({r["scene_id"] for r in rows})[: args.max_scenes]
        rows = [r for r in rows if r["scene_id"] in scenes]
        manifest_use = out / "manifest_subset.jsonl"
        write_jsonl(manifest_use, rows)

    steps = [
        [
            sys.executable,
            "-m",
            "volrecon.scripts.run_foundation_stereo",
            "--manifest",
            str(manifest_use),
            "--foundationstereo_repo",
            str(args.foundationstereo_repo),
            "--ckpt",
            str(args.ckpt),
            "--out",
            str(depth_out),
            "--max_views_per_scene",
            str(args.max_views_per_scene),
            "--scale",
            str(args.scale),
            "--valid_iters",
            str(args.valid_iters),
            "--min_depth_m",
            str(min_depth),
            "--max_depth_m",
            str(max_depth),
        ],
        [
            sys.executable,
            "-m",
            "volrecon.scripts.run_plain_tsdf",
            "--manifest",
            str(manifest_use),
            "--depth_predictions",
            str(depth_out),
            "--out",
            str(recon_out),
        ]
        + (["--config", str(args.config)] if args.config else []),
        [
            sys.executable,
            "-m",
            "volrecon.scripts.evaluate_reconstruction",
            "--manifest",
            str(manifest_use),
            "--recon_dir",
            str(recon_out),
            "--depth_predictions",
            str(depth_out),
            "--out",
            str(eval_out),
        ],
    ]

    for cmd in steps:
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    logger.info("End-to-end complete. Outputs in %s", out)


if __name__ == "__main__":
    main()

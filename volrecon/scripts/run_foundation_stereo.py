"""Run FoundationStereo depth prediction on manifest views."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from volrecon.config import PROJECT_ROOT
from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.io.json_io import read_jsonl
from volrecon.stereo.foundation_stereo_wrapper import (
    FoundationStereoConfig,
    FoundationStereoWrapper,
    NO_STEREO_ERROR,
    resolve_view_paths,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FoundationStereo on manifest stereo views.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--foundationstereo_repo", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max_views_per_scene", type=int, default=20)
    parser.add_argument("--min_depth_m", type=float, default=0.1)
    parser.add_argument("--max_depth_m", type=float, default=2.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--valid_iters", type=int, default=16)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    cfg = FoundationStereoConfig(
        foundationstereo_repo=args.foundationstereo_repo,
        ckpt=args.ckpt,
        scale=args.scale,
        valid_iters=args.valid_iters,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        project_root=args.project_root,
    )
    wrapper = FoundationStereoWrapper(cfg)

    records = [ViewRecord.from_dict(r) for r in read_jsonl(args.manifest)]
    by_scene: dict[str, list[ViewRecord]] = defaultdict(list)
    for r in records:
        by_scene[r.scene_id].append(r)

    for scene_id, views in sorted(by_scene.items()):
        views = sorted(views, key=lambda v: v.view_id)[: args.max_views_per_scene]
        for view in views:
            out_dir = args.out / scene_id / view.view_id
            if (out_dir / "depth_m.npy").exists():
                logger.info("Skipping existing %s/%s", scene_id, view.view_id)
                continue
            try:
                left, right = resolve_view_paths(view, args.project_root)
                wrapper.run_view(view, left, right, out_dir)
                logger.info("Saved depth prediction %s", out_dir)
            except ValueError as exc:
                if NO_STEREO_ERROR.split(".")[0] in str(exc):
                    logger.error("View %s/%s: %s", scene_id, view.view_id, exc)
                else:
                    raise


if __name__ == "__main__":
    main()

"""Compute per-view uncertainty / confidence maps."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import yaml

from volrecon.config import PROJECT_ROOT
from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.io.json_io import read_jsonl
from volrecon.uncertainty.calibration import UncertaintyConfig
from volrecon.uncertainty.uncertainty_model import compute_view_uncertainty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute uncertainty maps for depth predictions.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--depth_predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run_right_to_left", action="store_true")
    parser.add_argument("--tau_lr_px", type=float, default=1.5)
    parser.add_argument("--tau_photo", type=float, default=0.08)
    parser.add_argument("--tau_mv_m", type=float, default=0.005)
    parser.add_argument("--k_neighbor_views", type=int, default=5)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    cfg = UncertaintyConfig()
    if args.config and args.config.exists():
        with args.config.open("r", encoding="utf-8") as f:
            cfg = UncertaintyConfig.from_dict(yaml.safe_load(f))
    cfg.run_right_to_left = args.run_right_to_left or cfg.run_right_to_left
    cfg.thresholds.tau_lr_px = args.tau_lr_px
    cfg.thresholds.tau_photo = args.tau_photo
    cfg.thresholds.tau_mv_m = args.tau_mv_m
    cfg.k_neighbor_views = args.k_neighbor_views

    records = [ViewRecord.from_dict(r) for r in read_jsonl(args.manifest)]
    by_scene: dict[str, list[ViewRecord]] = defaultdict(list)
    for r in records:
        by_scene[r.scene_id].append(r)

    for scene_id, views in sorted(by_scene.items()):
        for view in views:
            pred_dir = args.depth_predictions / scene_id / view.view_id
            out_dir = args.out / scene_id / view.view_id
            if (out_dir / "confidence_total.npy").exists():
                continue
            if not (pred_dir / "depth_m.npy").exists():
                logger.warning("No depth for %s/%s", scene_id, view.view_id)
                continue

            disp_r2l = None
            if cfg.run_right_to_left and (pred_dir / "disparity_r2l.npy").exists():
                import numpy as np

                disp_r2l = np.load(pred_dir / "disparity_r2l.npy")

            compute_view_uncertainty(
                view,
                pred_dir,
                out_dir,
                cfg,
                scene_views=views,
                depth_pred_root=args.depth_predictions,
                project_root=args.project_root,
                disparity_r2l=disp_r2l,
            )
            logger.info("Uncertainty maps: %s/%s", scene_id, view.view_id)


if __name__ == "__main__":
    main()

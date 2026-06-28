"""Grid/random search over uncertainty hyperparameters."""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import random
from pathlib import Path

import yaml

from volrecon.config import PROJECT_ROOT
from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.eval.gt_builders import load_gt_scene_mesh
from volrecon.eval.reconstruction_metrics import compute_reconstruction_metrics
from volrecon.fusion.bounds import compute_bounds_from_depth_points, robust_expand_bounds
from volrecon.fusion.fusion_utils import world_cam_from_view
from volrecon.fusion.weighted_tsdf import DenseChunkedWeightedTSDF, WeightedTSDFConfig
from volrecon.io.json_io import read_json, read_jsonl, write_json
from volrecon.uncertainty.calibration import UncertaintyConfig
from volrecon.uncertainty.confidence_sources import compute_confidence_maps
from volrecon.io.image_io import read_rgb
from volrecon.stereo.foundation_stereo_wrapper import resolve_view_paths
import numpy as np
import trimesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_val_scenes(manifest: Path, allow_test: bool, max_scenes: int) -> dict[str, list[ViewRecord]]:
    records = [ViewRecord.from_dict(r) for r in read_jsonl(manifest)]
    if not allow_test:
        records = [r for r in records if r.split != "test"]
    scenes: dict[str, list[ViewRecord]] = {}
    for r in records:
        scenes.setdefault(r.scene_id, []).append(r)
    ids = sorted(scenes)[:max_scenes]
    return {k: scenes[k] for k in ids}


def _score_config(
    cfg_unc: UncertaintyConfig,
    cfg_tsdf: WeightedTSDFConfig,
    scenes: dict[str, list[ViewRecord]],
    depth_root: Path,
    project_root: Path,
) -> float:
    total = 0.0
    n = 0
    for scene_id, views in scenes.items():
        gt = load_gt_scene_mesh(scene_id, views[0].dataset, project_root)
        if gt is None:
            continue
        depth_maps, Ks, T_wcs, frames = [], [], [], []
        for view in views[:5]:
            pd = depth_root / scene_id / view.view_id
            if not (pd / "depth_m.npy").exists():
                continue
            depth = np.load(pd / "depth_m.npy")
            disp = np.load(pd / "disparity.npy")
            left, right = resolve_view_paths(view, project_root)
            K = np.asarray(read_json(pd / "K_scaled.json")["K"] if (pd / "K_scaled.json").exists() else view.K)
            maps = compute_confidence_maps(
                depth, disp, read_rgb(left), read_rgb(right), K, cfg_unc
            )
            T_wc, T_cw = world_cam_from_view(view, object_centric=view.dataset == "bop_tless")
            depth_maps.append(depth)
            Ks.append(K)
            T_wcs.append(T_wc)
            frames.append((depth, maps.weight_total, K, T_cw))

        if not frames:
            continue
        bounds = robust_expand_bounds(compute_bounds_from_depth_points(depth_maps, Ks, T_wcs), 0.05)
        tsdf = DenseChunkedWeightedTSDF(bounds, cfg_tsdf)
        for depth, weight, K, T_cw in frames:
            tsdf.integrate_view(depth, weight, K, T_cw)
        mesh = tsdf.extract_mesh()
        if len(mesh.faces) == 0:
            continue
        m = compute_reconstruction_metrics(mesh, gt, num_sample_points=20_000)
        total += m.chamfer_l1_m
        n += 1
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune uncertainty weights on validation scenes.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--depth_predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max_scenes", type=int, default=3)
    parser.add_argument("--allow_test_tuning", action="store_true")
    parser.add_argument("--random_trials", type=int, default=10)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    scenes = _load_val_scenes(args.manifest, args.allow_test_tuning, args.max_scenes)
    alphas = [0.5, 1.0, 2.0]
    taus = [1.0, 1.5, 2.0]

    results = []
    best_score = float("inf")
    best_cfg = None

    trials = list(itertools.product(alphas, taus))[: args.random_trials]
    random.shuffle(trials)

    for alpha_lr, tau_lr in trials:
        unc = UncertaintyConfig()
        unc.exponents.alpha_lr = alpha_lr
        unc.thresholds.tau_lr_px = tau_lr
        tsdf = WeightedTSDFConfig()
        score = _score_config(unc, tsdf, scenes, args.depth_predictions, args.project_root)
        row = {"alpha_lr": alpha_lr, "tau_lr_px": tau_lr, "val_chamfer_l1_m": score}
        results.append(row)
        logger.info("Trial %s -> %.5f", row, score)
        if score < best_score:
            best_score = score
            best_cfg = {"uncertainty": {"alpha_lr": alpha_lr, "tau_lr_px": tau_lr}, "score": score}

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "tuning_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["alpha_lr"])
        writer.writeheader()
        writer.writerows(results)

    if best_cfg:
        with (args.out / "best_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(best_cfg, f)
    write_json(args.out / "tuning_summary.json", {"best": best_cfg, "trials": len(results)})


if __name__ == "__main__":
    main()

"""Run weighted TSDF scene reconstruction."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from volrecon.config import PROJECT_ROOT
from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.fusion.bounds import compute_bounds_from_depth_points, robust_expand_bounds, save_bounds_json
from volrecon.fusion.fusion_utils import PoseConventionError, world_cam_from_view
from volrecon.fusion.plain_tsdf import depth_prediction_dir
from volrecon.fusion.weighted_tsdf import DenseChunkedWeightedTSDF, WeightedTSDFConfig
from volrecon.fusion.weighted_volume import compute_weighted_volumes
from volrecon.io.image_io import read_rgb
from volrecon.io.json_io import read_json, read_jsonl, write_json
from volrecon.stereo.foundation_stereo_wrapper import resolve_view_paths

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reconstruct_weighted_scene(
    scene_id: str,
    views: list[ViewRecord],
    depth_root: Path,
    uncertainty_root: Path,
    out_dir: Path,
    cfg: WeightedTSDFConfig,
    project_root: Path,
    object_centric: bool = False,
) -> Path:
    views = sorted(views, key=lambda v: v.view_id)
    depth_maps, Ks, T_wcs, frame_data = [], [], [], []

    for view in views:
        pred = depth_prediction_dir(depth_root, scene_id, view.view_id)
        unc = uncertainty_root / scene_id / view.view_id
        if not (pred / "depth_m.npy").exists() or not (unc / "weight_total.npy").exists():
            logger.warning("Skip view %s: missing depth or uncertainty", view.view_id)
            continue
        k_path = pred / "K_scaled.json"
        K = np.asarray(read_json(k_path)["K"] if k_path.exists() else view.K, dtype=np.float64)
        try:
            T_wc, T_cw = world_cam_from_view(view, object_centric=object_centric)
        except PoseConventionError as exc:
            logger.warning("Skip view %s: %s", view.view_id, exc)
            continue
        depth_m = np.load(pred / "depth_m.npy")
        weight = np.load(unc / "weight_total.npy")
        depth_maps.append(depth_m)
        Ks.append(K)
        T_wcs.append(T_wc)
        frame_data.append((view, depth_m, weight, K, T_cw))

    if not frame_data:
        raise PoseConventionError(f"No integrable views for weighted TSDF scene {scene_id}")

    bounds = robust_expand_bounds(compute_bounds_from_depth_points(depth_maps, Ks, T_wcs), 0.05)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_bounds_json(out_dir / "bounds.json", bounds, "depth_pointcloud")

    tsdf = DenseChunkedWeightedTSDF(bounds, cfg)
    for view, depth_m, weight, K, T_cw in frame_data:
        rgb = None
        try:
            lp, _ = resolve_view_paths(view, project_root)
            rgb = read_rgb(lp)
        except Exception:  # noqa: BLE001
            pass
        tsdf.integrate_view(depth_m, weight, K, T_cw, rgb=rgb, view_id=view.view_id)

    paths = tsdf.save_outputs(out_dir)
    vol = compute_weighted_volumes(
        paths["mesh_weighted_clean"],
        cfg.voxel_length_m,
        occupancy_path=out_dir / "occupancy_grid.npz" if cfg.use_occupancy else None,
        occupancy_threshold=cfg.occupancy_threshold,
        min_weight=cfg.min_weight_for_mesh,
    )
    write_json(out_dir / "volume.json", vol.to_dict())
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run uncertainty-weighted TSDF fusion.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--depth_predictions", required=True, type=Path)
    parser.add_argument("--uncertainty_dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--voxel_length_m", type=float, default=0.003)
    parser.add_argument("--sdf_trunc_m", type=float, default=0.015)
    parser.add_argument("--min_weight", type=float, default=2.0)
    parser.add_argument("--max_weight_per_obs", type=float, default=5.0)
    parser.add_argument("--W_max", type=float, default=100.0)
    parser.add_argument("--use_occupancy", action="store_true")
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    cfg = WeightedTSDFConfig(
        voxel_length_m=args.voxel_length_m,
        sdf_trunc_m=args.sdf_trunc_m,
        min_weight_for_mesh=args.min_weight,
        max_weight_per_obs=args.max_weight_per_obs,
        W_max=args.W_max,
        use_occupancy=args.use_occupancy,
    )
    if args.config and args.config.exists():
        with args.config.open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        integ = y.get("integration", y)
        cfg.voxel_length_m = y.get("voxel_length_m", cfg.voxel_length_m)
        cfg.sdf_trunc_m = y.get("sdf_trunc_m", cfg.sdf_trunc_m)
        cfg.min_depth_m = y.get("min_depth_m", cfg.min_depth_m)
        cfg.max_depth_m = y.get("max_depth_m", cfg.max_depth_m)
        cfg.min_weight_for_mesh = integ.get("min_weight_for_mesh", cfg.min_weight_for_mesh)
        cfg.W_max = integ.get("W_max", cfg.W_max)
        cfg.use_occupancy = integ.get("use_occupancy", cfg.use_occupancy)
        cfg.occupancy_threshold = integ.get("occupancy_threshold", cfg.occupancy_threshold)
        rob = y.get("uncertainty", {}).get("robust_kernel", y.get("robust_kernel", {}))
        cfg.robust_kernel = rob.get("type", cfg.robust_kernel)
        cfg.robust_delta = rob.get("delta", cfg.robust_delta)

    records = [ViewRecord.from_dict(r) for r in read_jsonl(args.manifest)]
    by_scene: dict[str, list[ViewRecord]] = defaultdict(list)
    for r in records:
        by_scene[r.scene_id].append(r)

    for scene_id, views in sorted(by_scene.items()):
        try:
            reconstruct_weighted_scene(
                scene_id,
                views,
                args.depth_predictions,
                args.uncertainty_dir,
                args.out / scene_id,
                cfg,
                args.project_root,
                object_centric=views[0].dataset == "bop_tless",
            )
            logger.info("Weighted TSDF done: %s", scene_id)
        except PoseConventionError as exc:
            logger.error("Scene %s: %s", scene_id, exc)


if __name__ == "__main__":
    main()

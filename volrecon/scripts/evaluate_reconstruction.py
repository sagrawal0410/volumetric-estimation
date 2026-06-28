"""Evaluate reconstructions against GT."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import trimesh

from volrecon.config import PROJECT_ROOT
from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.eval.gt_builders import (
    build_bop_union_gt_mesh,
    gt_volume_from_union_voxels,
    load_gt_scene_mesh,
    resolve_path,
)
from volrecon.eval.reconstruction_metrics import compute_depth_metrics, compute_reconstruction_metrics
from volrecon.eval.reports import summarize_run, write_scene_eval_bundle
from volrecon.eval.volume_metrics import compare_volumes
from volrecon.geometry.depth import depth_uint16_to_meters
from volrecon.geometry.mesh_volume import compute_mesh_volume_report, load_mesh_volume_report
from volrecon.io.image_io import read_depth_png
from volrecon.io.json_io import read_json, read_jsonl
from volrecon.visualization.depth_debug import save_depth_debug_grid
from volrecon.visualization.html_report import write_html_report, write_run_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate plain TSDF reconstructions.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--recon_dir", required=True, type=Path)
    parser.add_argument("--depth_predictions", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--num_sample_points", type=int, default=100_000)
    parser.add_argument("--thresholds_m", type=float, nargs="+", default=[0.001, 0.002, 0.005, 0.010])
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    records = [ViewRecord.from_dict(r) for r in read_jsonl(args.manifest)]
    scenes = sorted({r.scene_id for r in records})
    scene_reports = []

    for scene_id in scenes:
        recon_scene = args.recon_dir / scene_id
        pred_mesh_path = recon_scene / "mesh_clean.ply"
        if not pred_mesh_path.exists():
            logger.warning("No reconstruction for scene %s", scene_id)
            continue

        pred_mesh = trimesh.load(pred_mesh_path, force="mesh", process=False)
        dataset = next(r.dataset for r in records if r.scene_id == scene_id)

        gt_mesh = load_gt_scene_mesh(scene_id, dataset, args.project_root)
        gt_source = "scene_gt_mesh"
        if gt_mesh is None and dataset == "bop_tless":
            gt_mesh = build_bop_union_gt_mesh(scene_id, args.project_root)
            gt_source = "bop_union_meshes"

        scene_out = args.out / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)

        metrics_dict: dict = {"scene_id": scene_id}
        volume_dict: dict = {"scene_id": scene_id}

        if gt_mesh is not None:
            rec_m = compute_reconstruction_metrics(
                pred_mesh, gt_mesh, args.num_sample_points, args.thresholds_m
            )
            metrics_dict.update(rec_m.to_dict())
            vol_cmp, pred_vol, gt_vol = compare_volumes(pred_mesh, gt_mesh, gt_source=gt_source)
            volume_dict.update(vol_cmp.to_dict())
            volume_dict["predicted"] = pred_vol.to_dict()
            volume_dict["gt"] = gt_vol.to_dict()
        elif dataset == "bop_tless":
            gt_vol_m3 = gt_volume_from_union_voxels(scene_id, args.project_root)
            if gt_vol_m3 is not None:
                pred_vol = compute_mesh_volume_report(pred_mesh)
                abs_err = abs(pred_vol.volume_m3 - gt_vol_m3)
                volume_dict.update(
                    {
                        "predicted_volume_m3": pred_vol.volume_m3,
                        "gt_volume_m3": gt_vol_m3,
                        "abs_volume_error_m3": abs_err,
                        "rel_volume_error_percent": 100 * abs_err / max(gt_vol_m3, 1e-9),
                        "gt_source": "union_gt_voxels",
                    }
                )

        # Per-view depth eval (eval-only GT depth)
        depth_metrics_views = []
        if args.depth_predictions:
            for view in [r for r in records if r.scene_id == scene_id]:
                if not view.gt_depth_path:
                    continue
                pred_depth_path = args.depth_predictions / scene_id / view.view_id / "depth_m.npy"
                if not pred_depth_path.exists():
                    continue
                pred_d = np.load(pred_depth_path)
                gt_path = resolve_path(view.gt_depth_path, args.project_root)
                meta_path = gt_path.parent / "meta.json"
                scale = 1.0
                if meta_path.exists():
                    scale = float(read_json(meta_path).get("depth_scale", 1.0))
                gt_d = read_depth_png(gt_path, scale=scale)
                if gt_d.shape != pred_d.shape:
                    import cv2

                    gt_d = cv2.resize(gt_d, (pred_d.shape[1], pred_d.shape[0]), interpolation=cv2.INTER_NEAREST)
                dm = compute_depth_metrics(pred_d, gt_d)
                dm["view_id"] = view.view_id
                depth_metrics_views.append(dm)
            metrics_dict["depth_per_view"] = depth_metrics_views

        write_scene_eval_bundle(scene_out, metrics_dict, volume_dict)

        img_paths = {"depth_debug": scene_out / "depth_debug_grid.png"}
        if depth_metrics_views and args.depth_predictions:
            v0 = next(r for r in records if r.scene_id == scene_id and r.gt_depth_path)
            pd = np.load(args.depth_predictions / scene_id / v0.view_id / "depth_m.npy")
            gt_p = resolve_path(v0.gt_depth_path, args.project_root)
            meta_path = gt_p.parent / "meta.json"
            scale = float(read_json(meta_path).get("depth_scale", 1.0)) if meta_path.exists() else 1.0
            gt_d = read_depth_png(gt_p, scale=scale)
            save_depth_debug_grid(scene_out / "depth_debug_grid.png", None, None, pd, gt_d)
            img_paths["depth_debug"] = scene_out / "depth_debug_grid.png"

        write_html_report(
            scene_out / "report.html",
            scene_id,
            metrics_dict,
            volume_dict,
            img_paths,
        )
        scene_reports.append({"scene_id": scene_id, "metrics": metrics_dict, "volume": volume_dict})

    summary = summarize_run(scene_reports)
    write_run_summary(args.out, summary)
    logger.info("Evaluation complete: %s", args.out)


if __name__ == "__main__":
    main()

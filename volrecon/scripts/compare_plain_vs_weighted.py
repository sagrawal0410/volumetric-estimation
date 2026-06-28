"""Compare plain vs weighted TSDF reconstructions."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import trimesh
import yaml

from volrecon.config import PROJECT_ROOT
from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.eval.gt_builders import build_bop_union_gt_mesh, load_gt_scene_mesh
from volrecon.eval.reconstruction_metrics import compute_depth_metrics, compute_reconstruction_metrics
from volrecon.eval.volume_metrics import compare_volumes
from volrecon.fusion.weighted_volume import compute_weighted_volumes
from volrecon.geometry.depth import depth_uint16_to_meters
from volrecon.io.image_io import read_depth_png
from volrecon.io.json_io import read_json, read_jsonl, write_json
from volrecon.visualization.depth_debug import save_depth_debug_grid
from volrecon.visualization.html_report import write_html_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def error_rejection_curve(pred: np.ndarray, gt: np.ndarray, confidence: np.ndarray, fractions: list[float]) -> list[dict]:
    valid = (pred > 0) & (gt > 0) & np.isfinite(pred) & np.isfinite(gt)
    if not np.any(valid):
        return []
    p = pred[valid]
    g = gt[valid]
    c = confidence[valid]
    order = np.argsort(-c)
    curves = []
    for frac in fractions:
        n = max(int(len(order) * frac), 1)
        sel = order[:n]
        m = compute_depth_metrics(p[sel], g[sel], valid_mask=np.ones(len(sel), dtype=bool))
        m["retained_fraction"] = frac
        curves.append(m)
    return curves


def eval_scene(
    scene_id: str,
    views: list[ViewRecord],
    plain_dir: Path,
    weighted_dir: Path,
    depth_pred_dir: Path | None,
    uncertainty_dir: Path | None,
    num_sample_points: int,
    thresholds_m: list[float],
    project_root: Path,
) -> dict:
    dataset = views[0].dataset
    gt_mesh = load_gt_scene_mesh(scene_id, dataset, project_root)
    if gt_mesh is None and dataset == "bop_tless":
        gt_mesh = build_bop_union_gt_mesh(scene_id, project_root)

    plain_mesh_path = plain_dir / scene_id / "mesh_clean.ply"
    weighted_mesh_path = weighted_dir / scene_id / "mesh_weighted_clean.ply"
    report: dict = {"scene_id": scene_id}

    if plain_mesh_path.exists() and gt_mesh is not None:
        plain_mesh = trimesh.load(plain_mesh_path, force="mesh", process=False)
        report["plain_reconstruction"] = compute_reconstruction_metrics(
            plain_mesh, gt_mesh, num_sample_points, thresholds_m
        ).to_dict()

    if weighted_mesh_path.exists() and gt_mesh is not None:
        weighted_mesh = trimesh.load(weighted_mesh_path, force="mesh", process=False)
        report["weighted_reconstruction"] = compute_reconstruction_metrics(
            weighted_mesh, gt_mesh, num_sample_points, thresholds_m
        ).to_dict()

    if plain_mesh_path.exists() and gt_mesh is not None:
        report["plain_volume"] = compare_volumes(
            trimesh.load(plain_mesh_path, force="mesh"), gt_mesh
        )[0].to_dict()

    if weighted_mesh_path.exists():
        wvol = compute_weighted_volumes(
            weighted_mesh_path,
            0.003,
            occupancy_path=weighted_dir / scene_id / "occupancy_grid.npz",
        )
        report["weighted_volume"] = wvol.to_dict()
        if gt_mesh is not None:
            report["weighted_volume_compare"] = compare_volumes(
                trimesh.load(weighted_mesh_path, force="mesh"), gt_mesh
            )[0].to_dict()

    depth_eval = []
    rejection_curves = []
    if depth_pred_dir and uncertainty_dir:
        for view in views:
            if not view.gt_depth_path:
                continue
            pd = depth_pred_dir / scene_id / view.view_id / "depth_m.npy"
            ud = uncertainty_dir / scene_id / view.view_id / "confidence_total.npy"
            if not pd.exists():
                continue
            pred = np.load(pd)
            conf = np.load(ud) if ud.exists() else np.ones_like(pred)
            gt_p = Path(view.gt_depth_path)
            if not gt_p.is_absolute():
                gt_p = project_root / gt_p
            meta = gt_p.parent / "meta.json"
            scale = float(read_json(meta).get("depth_scale", 1.0)) if meta.exists() else 1.0
            gt = read_depth_png(gt_p, scale=scale)
            if gt.shape != pred.shape:
                import cv2

                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            dm = compute_depth_metrics(pred, gt)
            dm["view_id"] = view.view_id
            dm["confidence_error_corr"] = float(np.corrcoef(conf[pred > 0].ravel(), np.abs(pred - gt)[pred > 0].ravel())[0, 1]) if np.any(pred > 0) else float("nan")
            depth_eval.append(dm)
            rejection_curves.extend(error_rejection_curve(pred, gt, conf, [0.25, 0.5, 0.75, 1.0]))

    report["depth_per_view"] = depth_eval
    report["error_rejection_curve"] = rejection_curves

    scene_out = args.out / scene_id
    scene_out.mkdir(parents=True, exist_ok=True)
    write_json(scene_out / "metrics.json", report)

    if depth_pred_dir and views:
        v0 = views[0]
        pd = depth_pred_dir / scene_id / v0.view_id / "depth_m.npy"
        if pd.exists() and v0.gt_depth_path:
            gt_p = project_root / v0.gt_depth_path if not Path(v0.gt_depth_path).is_absolute() else Path(v0.gt_depth_path)
            if gt_p.exists():
                save_depth_debug_grid(scene_out / "depth_debug_grid.png", None, None, np.load(pd), read_depth_png(gt_p))

    write_html_report(
        scene_out / "report.html",
        scene_id,
        report.get("weighted_reconstruction", report.get("plain_reconstruction", {})),
        report.get("weighted_volume_compare", report.get("plain_volume", {})),
        {"depth_debug": scene_out / "depth_debug_grid.png"},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare plain vs weighted TSDF.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--plain_dir", required=True, type=Path)
    parser.add_argument("--weighted_dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--depth_predictions", type=Path, default=None)
    parser.add_argument("--uncertainty_dir", type=Path, default=None)
    parser.add_argument("--num_sample_points", type=int, default=100_000)
    parser.add_argument("--thresholds_m", type=float, nargs="+", default=[0.001, 0.002, 0.005, 0.010])
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    records = [ViewRecord.from_dict(r) for r in read_jsonl(args.manifest)]
    scenes: dict[str, list[ViewRecord]] = {}
    for r in records:
        scenes.setdefault(r.scene_id, []).append(r)

    rows = []
    all_reports = []
    for scene_id, views in sorted(scenes.items()):
        rep = eval_scene(
            scene_id,
            views,
            args.plain_dir,
            args.weighted_dir,
            args.depth_predictions,
            args.uncertainty_dir,
            args.num_sample_points,
            args.thresholds_m,
            args.project_root,
        )
        all_reports.append(rep)
        plain_c = rep.get("plain_reconstruction", {}).get("chamfer_l1_m")
        weighted_c = rep.get("weighted_reconstruction", {}).get("chamfer_l1_m")
        rows.append(
            {
                "scene_id": scene_id,
                "plain_chamfer_l1_m": plain_c,
                "weighted_chamfer_l1_m": weighted_c,
                "plain_rel_vol_err_pct": rep.get("plain_volume", {}).get("rel_volume_error_percent"),
                "weighted_rel_vol_err_pct": rep.get("weighted_volume_compare", {}).get("rel_volume_error_percent"),
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "comparison_summary.json", {"scenes": all_reports})

    with (args.out / "aggregate_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    logger.info("Comparison report written to %s", args.out)


if __name__ == "__main__":
    main()

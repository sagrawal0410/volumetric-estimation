"""Batch prepare and evaluate WildRGB-D scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from wildrgbd_volume_benchmark.io_wildrgbd import discover_scenes
from wildrgbd_volume_benchmark.prepare_scene import prepare_scene
from wildrgbd_volume_benchmark.run_eval import run_eval
from wildrgbd_volume_benchmark.subset_sampler import (
    build_category_scene_manifest,
    materialize_subset,
    sample_scenes_per_category,
)


def _plot_batch(df: pd.DataFrame, out_root: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots = out_root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    valid = df.dropna(subset=["relative_error_percent"])
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    valid.groupby("method")["relative_error_percent"].mean().plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean rel error (%)")
    ax.set_title("Error by method")
    fig.tight_layout()
    fig.savefig(plots / "error_by_method.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    valid.groupby("category")["relative_error_percent"].mean().plot(kind="bar", ax=ax)
    ax.set_title("Error by category")
    fig.tight_layout()
    fig.savefig(plots / "error_by_category.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(valid["pseudo_gt_cm3"], valid["pred_cm3"], alpha=0.6)
    lims = [min(valid["pseudo_gt_cm3"].min(), valid["pred_cm3"].min()),
            max(valid["pseudo_gt_cm3"].max(), valid["pred_cm3"].max())]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlabel("Pseudo-GT cm³")
    ax.set_ylabel("Predicted cm³")
    fig.tight_layout()
    fig.savefig(plots / "pred_vs_pseudo_gt.png", dpi=150)
    plt.close(fig)

    fail = df.assign(failed=df["status"] != "ok").groupby("category")["failed"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    fail.plot(kind="bar", ax=ax)
    ax.set_ylabel("Failure rate")
    fig.tight_layout()
    fig.savefig(plots / "failure_rate_by_category.png", dpi=150)
    plt.close(fig)


def run_batch(
    wildrgbd_root: str | Path,
    out_root: str | Path,
    scene_types: tuple[str, ...] = ("single",),
    num_views: int = 5,
    methods: list[str] | None = None,
    categories: list[str] | None = None,
    samples_per_category: int | None = None,
) -> Path:
    methods = methods or ["convex_hull", "tsdf", "voxel_carving"]
    root = Path(wildrgbd_root).resolve()
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if samples_per_category is not None:
        manifest = build_category_scene_manifest(root, categories=categories, output_csv=out_root / "manifest.csv")
        selected = sample_scenes_per_category(manifest, samples_per_category=samples_per_category, scene_types=scene_types)
        scenes_info = selected
    else:
        scenes = discover_scenes(root, categories=categories, scene_types=scene_types)
        scenes_info = [
            {"category": s.category, "scene_id": s.scene_id, "scene_dir": s.scene_dir, "scene_type": s.scene_type}
            for s in scenes
        ]

    rows = []
    for info in tqdm(scenes_info, desc="scenes"):
        cat = info["category"]
        sid = info["scene_id"]
        prep_dir = out_root / "prepared" / cat / sid
        try:
            prepare_scene(
                wildrgbd_root=root,
                category=cat,
                scene_id=sid,
                out_dir=prep_dir,
                num_views=num_views,
                scene_types=scene_types,
            )
            reports = run_eval(prep_dir, methods)
            with (prep_dir / "pseudo_gt" / "pseudo_gt_volume.json").open("r", encoding="utf-8") as f:
                pg = json.load(f)
            with (prep_dir / "sampled_5view" / "selected_views.json").open("r", encoding="utf-8") as f:
                sv = json.load(f)
        except Exception as exc:
            for m in methods:
                rows.append({
                    "category": cat, "scene_id": sid, "method": m,
                    "status": f"failed: {exc}", "scene_type": info.get("scene_type"),
                })
            continue

        for r in reports:
            rows.append({
                "category": cat,
                "scene_id": sid,
                "scene_type": info.get("scene_type"),
                "num_full_frames_used": pg.get("num_full_frames_used"),
                "num_sampled_views": sv.get("num_views"),
                "pseudo_gt_cm3": pg.get("volume_cm3"),
                "method": r["method"],
                "pred_cm3": r.get("volume_cm3"),
                "relative_error_percent": r.get("relative_error_percent"),
                "status": r.get("status"),
                "tsdf_watertight": r.get("watertight") if r["method"] == "tsdf" else None,
            })

    df = pd.DataFrame(rows)
    agg_path = out_root / "aggregate_results.csv"
    df.to_csv(agg_path, index=False)
    if not df.empty and "relative_error_percent" in df:
        df.dropna(subset=["relative_error_percent"]).groupby("category")["relative_error_percent"].mean().to_csv(
            out_root / "category_summary.csv"
        )
        df.dropna(subset=["relative_error_percent"]).groupby("method")["relative_error_percent"].mean().to_csv(
            out_root / "method_summary.csv"
        )
    _plot_batch(df, out_root)
    print(f"Aggregate results: {agg_path}")
    return agg_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Batch WildRGB-D volume benchmark")
    parser.add_argument("--wildrgbd_root", required=True)
    parser.add_argument("--out_root", default="experiments/wildrgbd_subset_5view")
    parser.add_argument("--samples_per_category", type=int, default=None)
    parser.add_argument("--scene_types", default="single")
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--categories", default=None, help="Comma-separated category names")
    parser.add_argument("--methods", nargs="+", default=["convex_hull", "tsdf", "voxel_carving"])
    args = parser.parse_args(argv)
    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    types = tuple(t.strip() for t in args.scene_types.split(",") if t.strip())
    run_batch(
        wildrgbd_root=args.wildrgbd_root,
        out_root=args.out_root,
        scene_types=types,
        num_views=args.num_views,
        methods=args.methods,
        categories=cats,
        samples_per_category=args.samples_per_category,
    )


if __name__ == "__main__":
    main()

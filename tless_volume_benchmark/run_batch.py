"""Batch prepare + evaluate T-LESS objects."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from tless_volume_benchmark.run_eval import run_eval
from tless_volume_benchmark.tless_prepare import prepare_tless_scan


def _parse_object_ids(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _plot_batch(df: pd.DataFrame, out_root: Path) -> None:
    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    method_err = df.groupby("method")["rel_error_percent"].mean().dropna()
    method_err.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("Mean relative error (%)")
    ax.set_title("Error by method")
    ax.set_xlabel("Method")
    fig.tight_layout()
    fig.savefig(plots_dir / "error_by_method.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["gt_volume_cm3"], df["pred_volume_cm3"], alpha=0.6, c="coral")
    lims = [
        min(df["gt_volume_cm3"].min(), df["pred_volume_cm3"].min()),
        max(df["gt_volume_cm3"].max(), df["pred_volume_cm3"].max()),
    ]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlabel("GT volume (cm³)")
    ax.set_ylabel("Predicted volume (cm³)")
    ax.set_title("Predicted vs GT")
    fig.tight_layout()
    fig.savefig(plots_dir / "pred_vs_gt.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    obj_err = df.groupby("object_id")["rel_error_percent"].mean().dropna()
    obj_err.plot(kind="bar", ax=ax, color="seagreen")
    ax.set_ylabel("Mean relative error (%)")
    ax.set_title("Error by object")
    ax.set_xlabel("Object ID")
    fig.tight_layout()
    fig.savefig(plots_dir / "error_by_object.png", dpi=150)
    plt.close(fig)


def run_batch(
    dataset_root: str | Path,
    split: str,
    object_ids: list[int],
    out_root: str | Path,
    num_views: int = 5,
    min_visib_fract: float = 0.85,
    methods: list[str] | None = None,
) -> Path:
    methods = methods or ["convex_hull", "tsdf", "voxel_carving"]
    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for obj_id in tqdm(object_ids, desc="objects"):
        scan_name = f"tless_obj_{obj_id:06d}_{split}"
        scan_dir = out_root / "prepared" / scan_name
        try:
            prepare_tless_scan(
                dataset_root=dataset_root,
                split=split,
                object_id=obj_id,
                out_dir=scan_dir,
                num_views=num_views,
                min_visib_fract=min_visib_fract,
            )
            reports = run_eval(scan_dir, methods)
        except Exception as exc:
            for m in methods:
                rows.append(
                    {
                        "object_id": obj_id,
                        "method": m,
                        "status": f"failed: {exc}",
                        "split": split,
                        "num_views": num_views,
                    }
                )
            continue

        import json

        with (scan_dir / "selected_views.json").open("r", encoding="utf-8") as f:
            sv = json.load(f)
        views = sv.get("views", [])
        mean_visib = (
            sum(v["visib_fract"] for v in views if v.get("visib_fract") is not None)
            / max(1, sum(1 for v in views if v.get("visib_fract") is not None))
        )
        mean_valid = sum(v.get("valid_object_depth_pixels", 0) for v in views) / max(len(views), 1)

        with (scan_dir / "gt_volume.json").open("r", encoding="utf-8") as f:
            gt = json.load(f)

        for r in reports:
            rows.append(
                {
                    "object_id": obj_id,
                    "gt_volume_cm3": gt.get("volume_cm3"),
                    "method": r["method"],
                    "pred_volume_cm3": r.get("volume_cm3"),
                    "rel_error_percent": r.get("relative_error_percent"),
                    "split": split,
                    "num_views": len(views),
                    "mean_visib_fract": mean_visib,
                    "mean_valid_depth_pixels": mean_valid,
                    "tsdf_watertight": r.get("watertight") if r["method"] == "tsdf" else None,
                    "status": r.get("status"),
                }
            )

    summary_path = out_root / "aggregate_summary.csv"
    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)
    _plot_batch(df.dropna(subset=["rel_error_percent"]), out_root)
    print(f"Aggregate summary: {summary_path}")
    return summary_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Batch T-LESS volume benchmark")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--split", default="train_primesense")
    parser.add_argument("--object_ids", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--min_visib_fract", type=float, default=0.85)
    parser.add_argument("--out_root", default="experiments/tless_train_primesense")
    parser.add_argument("--methods", nargs="+", default=["convex_hull", "tsdf", "voxel_carving"])
    args = parser.parse_args(argv)
    run_batch(
        dataset_root=args.dataset_root,
        split=args.split,
        object_ids=_parse_object_ids(args.object_ids),
        out_root=args.out_root,
        num_views=args.num_views,
        min_visib_fract=args.min_visib_fract,
        methods=args.methods,
    )


if __name__ == "__main__":
    main()

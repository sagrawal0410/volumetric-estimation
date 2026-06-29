"""Evaluate pipeline against ground-truth CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console

from rtabmap_volume.eval.gt_compare import build_evaluation_table, compare_to_gt, save_evaluation_report
from rtabmap_volume.eval.metrics import summarize_evaluation
from rtabmap_volume.pipeline import run_pipeline

console = Console()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate volume estimates against ground truth")
    p.add_argument("--csv", required=True, help="CSV: scene_id,input_path,gt_volume_m3,config_path,units,roi_json")
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    df_in = pd.read_csv(args.csv)
    out = Path(args.out)
    rows = []

    for _, row in df_in.iterrows():
        scene_out = out / str(row["scene_id"])
        result = run_pipeline(
            input_path=row["input_path"],
            out_dir=scene_out,
            config_path=row["config_path"],
            units=row.get("units", "m"),
            roi_json=row.get("roi_json") if pd.notna(row.get("roi_json")) else None,
            overwrite=args.overwrite,
            command=f"evaluate {row['scene_id']}",
        )
        pred = result.get("final_volume_m3", 0.0) or 0.0
        gt = float(row["gt_volume_m3"])
        cmp = compare_to_gt(pred, gt)
        cmp["scene_id"] = row["scene_id"]
        cmp["confidence"] = result.get("confidence")
        rows.append(cmp)

    eval_df = build_evaluation_table(rows)
    summary = save_evaluation_report(eval_df, out)
    console.print(f"[green]Evaluation complete.[/green] MARE={summary['mare_percent']:.2f}% RMSE={summary['rmse_m3']:.4f} m³")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

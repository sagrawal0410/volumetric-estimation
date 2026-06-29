"""Ground truth comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rtabmap_volume.eval.metrics import absolute_error, relative_error_percent, summarize_evaluation


def compare_to_gt(predicted_m3: float, gt_m3: float) -> dict[str, float]:
    return {
        "predicted_volume_m3": predicted_m3,
        "gt_volume_m3": gt_m3,
        "absolute_error_m3": absolute_error(predicted_m3, gt_m3),
        "relative_error_percent": relative_error_percent(predicted_m3, gt_m3),
    }


def build_evaluation_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def save_evaluation_report(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "evaluation_results.csv", index=False)
    summary = summarize_evaluation(df)
    return summary

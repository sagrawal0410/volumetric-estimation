"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def absolute_error(pred: float, gt: float) -> float:
    return abs(pred - gt)


def relative_error_percent(pred: float, gt: float) -> float:
    if gt == 0:
        return float("inf") if pred != 0 else 0.0
    return abs(pred - gt) / abs(gt) * 100.0


def mean_absolute_relative_error(preds: np.ndarray, gts: np.ndarray) -> float:
    mask = gts != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(preds[mask] - gts[mask]) / np.abs(gts[mask])) * 100)


def rmse(preds: np.ndarray, gts: np.ndarray) -> float:
    return float(np.sqrt(np.mean((preds - gts) ** 2)))


def summarize_evaluation(df: pd.DataFrame) -> dict[str, float]:
    preds = df["predicted_volume_m3"].values
    gts = df["gt_volume_m3"].values
    return {
        "mare_percent": mean_absolute_relative_error(preds, gts),
        "rmse_m3": rmse(preds, gts),
        "n_scenes": len(df),
    }

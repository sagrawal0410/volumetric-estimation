"""Evaluation metrics and report row construction."""

from __future__ import annotations

from typing import Any


def relative_error_percent(pred_m3: float, gt_m3: float) -> float:
    """Relative error as a percentage: 100 * |pred - gt| / gt."""
    if gt_m3 <= 0:
        raise ValueError(f"Ground-truth volume must be positive, got {gt_m3}")
    return 100.0 * abs(pred_m3 - gt_m3) / gt_m3


def absolute_error_cm3(pred_m3: float, gt_m3: float) -> float:
    """Absolute error in cubic centimeters."""
    return abs(pred_m3 - gt_m3) * 1e6


def make_report_row(
    method: str,
    pred_m3: float,
    gt_m3: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single evaluation result row suitable for CSV/JSON export."""
    meta = dict(metadata or {})
    row: dict[str, Any] = {
        "method": method,
        "pred_volume_m3": float(pred_m3),
        "pred_volume_cm3": float(pred_m3 * 1e6),
        "gt_volume_m3": float(gt_m3),
        "gt_volume_cm3": float(gt_m3 * 1e6),
        "abs_error_cm3": absolute_error_cm3(pred_m3, gt_m3),
        "rel_error_percent": relative_error_percent(pred_m3, gt_m3),
    }
    row.update(meta)
    return row

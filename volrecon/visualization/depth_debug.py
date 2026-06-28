"""Depth debug visualizations."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def depth_error_heatmap(pred_m: np.ndarray, gt_m: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    if valid is None:
        valid = (pred_m > 0) & (gt_m > 0)
    err = np.zeros_like(pred_m, dtype=np.float64)
    err[valid] = np.abs(pred_m[valid] - gt_m[valid])
    if not np.any(valid):
        return (err * 255).astype(np.uint8)
    vmax = np.percentile(err[valid], 95)
    norm = np.clip(err / max(vmax, 1e-6), 0, 1)
    cm = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(cm, cv2.COLORMAP_JET)


def save_depth_debug_grid(
    out_path: Path,
    left: np.ndarray | None,
    right: np.ndarray | None,
    pred_depth: np.ndarray,
    gt_depth: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> None:
    panels = []
    titles = []

    if left is not None:
        panels.append(left if left.ndim == 3 else cv2.cvtColor(left, cv2.COLOR_GRAY2RGB))
        titles.append("left")
    if right is not None:
        panels.append(right if right.ndim == 3 else cv2.cvtColor(right, cv2.COLOR_GRAY2RGB))
        titles.append("right")

    pred_vis = pred_depth.copy()
    if valid_mask is not None:
        pred_vis = pred_vis.copy()
        pred_vis[~valid_mask] = 0
    if np.any(pred_vis > 0):
        d = pred_vis[pred_vis > 0]
        norm = (pred_vis - d.min()) / max(d.max() - d.min(), 1e-6)
        pred_rgb = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    else:
        pred_rgb = np.zeros((*pred_depth.shape, 3), dtype=np.uint8)
    panels.append(pred_rgb)
    titles.append("pred depth")

    if gt_depth is not None:
        valid = (gt_depth > 0) & (pred_depth > 0)
        panels.append(depth_error_heatmap(pred_depth, gt_depth, valid))
        titles.append("depth error")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, panels, titles, strict=True):
        ax.imshow(img if img.ndim == 3 else img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

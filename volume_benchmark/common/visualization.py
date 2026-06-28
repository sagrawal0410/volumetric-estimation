"""Optional visualization helpers for debugging scans and results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_depth_and_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path | None = None,
    title: str = "Depth + mask",
) -> None:
    """Save or show a side-by-side depth and mask visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    depth_show = depth_m.copy()
    depth_show[~mask] = np.nan
    im0 = axes[0].imshow(depth_show, cmap="viridis")
    axes[0].set_title("Depth (m)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    axes[1].imshow(mask.astype(np.uint8) * 255, cmap="gray")
    axes[1].set_title("Mask")
    fig.suptitle(title)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_error_bars(
    methods: list[str],
    rel_errors: list[float],
    out_path: str | Path | None = None,
    title: str = "Relative volume error (%)",
) -> None:
    """Bar chart of relative errors per method."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(methods))
    ax.bar(x, rel_errors)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

"""Debug visualization for WildRGB-D volume benchmark."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import trimesh

from wildrgbd_volume_benchmark.io_wildrgbd import WildRGBDFrame


def depth_to_vis(depth_m: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if mask is not None:
        valid &= mask
    vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return vis
    lo = float(np.percentile(depth_m[valid], 5))
    hi = float(np.percentile(depth_m[valid], 95))
    if hi <= lo:
        hi = lo + 0.01
    norm = np.clip((depth_m - lo) / (hi - lo), 0, 1)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    return vis


def visualize_depth_mask_overlay(rgb: np.ndarray, mask: np.ndarray, path: Path) -> None:
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    color = np.zeros_like(overlay)
    color[mask] = (0, 255, 0)
    blended = cv2.addWeighted(overlay, 0.7, color, 0.3, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), blended)


def visualize_scene_coverage(
    full_points: np.ndarray,
    camera_centers: list[np.ndarray],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if full_points.shape[0] > 5000:
        idx = np.random.choice(full_points.shape[0], 5000, replace=False)
        pts = full_points[idx]
    else:
        pts = full_points
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.2, c="gray", alpha=0.3)
    centers = np.array(camera_centers)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=40, c="red")
    ax.set_title("Camera centers vs full point cloud")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def visualize_fused_pointcloud(full_path: Path, sampled_path: Path, out_path: Path) -> None:
    full = trimesh.load(full_path)
    sampled = trimesh.load(sampled_path)
    combined = trimesh.util.concatenate([full, sampled]) if isinstance(sampled, trimesh.PointCloud) else full
    combined.export(out_path)


def save_frame_debug(frame: WildRGBDFrame, debug_dir: Path, prefix: str) -> None:
    assert frame.rgb is not None and frame.depth_m is not None and frame.mask is not None
    debug_dir.mkdir(parents=True, exist_ok=True)
    visualize_depth_mask_overlay(frame.rgb, frame.mask, debug_dir / f"{prefix}_mask_overlay.png")
    cv2.imwrite(str(debug_dir / f"{prefix}_depth_vis.png"), depth_to_vis(frame.depth_m, frame.mask))

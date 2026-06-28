"""Visualization helpers for ZED live captures."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from volrecon.io.json_io import read_jsonl


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_left_right_grid(scene_dir: Path, max_views: int = 6) -> Path:
    views_dir = scene_dir / "views"
    view_ids = sorted(p.name for p in views_dir.iterdir() if p.is_dir())[:max_views]
    tiles = []
    for vid in view_ids:
        left = _load_rgb(views_dir / vid / "left.png")
        right = _load_rgb(views_dir / vid / "right.png")
        tiles.append(np.hstack([left, right]))
    if not tiles:
        grid = np.zeros((64, 128, 3), dtype=np.uint8)
    else:
        grid = np.vstack(tiles)
    out = scene_dir / "left_right_grid.png"
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return out


def write_keyframe_contact_sheet(scene_dir: Path, cols: int = 4, max_views: int = 16) -> Path:
    views_dir = scene_dir / "views"
    view_ids = sorted(p.name for p in views_dir.iterdir() if p.is_dir())[:max_views]
    thumbs = []
    for vid in view_ids:
        img = _load_rgb(views_dir / vid / "left.png")
        thumbs.append(cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA))
    if not thumbs:
        out = scene_dir / "keyframe_contact_sheet.png"
        cv2.imwrite(str(out), np.zeros((180, 320, 3), dtype=np.uint8))
        return out
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    out = scene_dir / "keyframe_contact_sheet.png"
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return out


def write_camera_trajectory(scene_dir: Path) -> Path | None:
    positions = []
    for row in read_jsonl(scene_dir / "manifest.jsonl"):
        T = row.get("T_world_cam")
        if T is not None:
            positions.append(np.asarray(T, dtype=np.float64).reshape(4, 4)[:3, 3])
    if len(positions) < 2:
        return None
    pts = np.stack(positions, axis=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pts[:, 0], pts[:, 1], "o-", label="camera path (XY)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("ZED camera trajectory")
    ax.axis("equal")
    ax.legend()
    out = scene_dir / "camera_trajectory.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def write_mesh_preview(recon_out: Path, weighted: bool) -> Path | None:
    mesh_name = "mesh_weighted_clean.ply" if weighted else "mesh_clean.ply"
    for sd in recon_out.iterdir() if recon_out.exists() else []:
        mesh_path = sd / mesh_name
        if mesh_path.exists():
            return mesh_path
    return None


def write_zed_scene_visualizations(scene_dir: Path, recon_out: Path, weighted: bool = False) -> dict[str, Path | None]:
    outputs: dict[str, Path | None] = {}
    outputs["left_right_grid"] = write_left_right_grid(scene_dir)
    outputs["keyframe_contact_sheet"] = write_keyframe_contact_sheet(scene_dir)
    outputs["camera_trajectory"] = write_camera_trajectory(scene_dir)
    mesh_src = write_mesh_preview(recon_out, weighted)
    if mesh_src is not None:
        outputs["mesh_preview"] = mesh_src
    return outputs

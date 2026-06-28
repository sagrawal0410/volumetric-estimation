"""YCB rendered stereo preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from volume_benchmark.common.geometry import make_T
from volume_benchmark.common.mesh_volume import compute_mesh_volume_m3, load_mesh_as_meters
from volume_benchmark.stereo.render_stereo_from_mesh import render_rectified_stereo_from_mesh
from volume_benchmark.stereo.stereo_dataset_adapter import save_prepared_stereo_scan


def prepare_ycb_stereo_rendered(
    object_root: str | Path,
    out_dir: str | Path,
    baseline_m: float = 0.12,
    num_views: int = 5,
    image_size: tuple[int, int] = (640, 480),
) -> Path:
    """Render canonical orbital views from YCB mesh."""
    root = Path(object_root).resolve()
    mesh_files = list(root.glob("**/*.obj")) + list(root.glob("**/*.ply"))
    if not mesh_files:
        raise FileNotFoundError(f"No mesh found under {object_root}")
    mesh_path = mesh_files[0]

    mesh = load_mesh_as_meters(mesh_path, source_units="auto")
    vol_m3, watertight, gt_type = compute_mesh_volume_m3(mesh, repair=False)
    if vol_m3 is None:
        vol_m3 = abs(float(mesh.convex_hull.volume))
        gt_type = "mesh_convex_hull_fallback_not_exact"
        exact = False
    else:
        exact = gt_type == "mesh_watertight"

    width, height = image_size
    K = np.array([[600.0, 0, width / 2], [0, 600.0, height / 2], [0, 0, 1.0]])
    gt_volume = {
        "volume_m3": vol_m3,
        "volume_cm3": vol_m3 * 1e6,
        "gt_type": gt_type,
        "watertight": bool(watertight),
        "exact_gt": exact,
        "source_mesh": str(mesh_path),
    }

    frames_data = []
    radius = 0.5
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        eye = np.array([radius * np.cos(angle), 0.05, radius * np.sin(angle)])
        fwd = -eye / np.linalg.norm(eye)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right) + 1e-12
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd], axis=0)
        T = make_T(R, -R @ eye)
        left, right_img, mask, rmeta = render_rectified_stereo_from_mesh(
            mesh_path, K, (width, height), T, baseline_m, mesh_units="m"
        )
        fmeta = {**rmeta, "source_mode": "rendered_stereo_from_ycb_mesh", "view_index": i}
        frames_data.append((left, right_img, mask if mask is not None else np.ones((height, width), bool), T, fmeta))

    metadata = {
        "source_mode": "rendered_stereo_from_ycb_mesh",
        "object_root": str(root),
        "baseline_m": baseline_m,
    }
    return save_prepared_stereo_scan(out_dir, K, baseline_m, frames_data, mesh_path, gt_volume, metadata=metadata)

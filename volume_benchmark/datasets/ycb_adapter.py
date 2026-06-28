"""Adapter for YCB-Video / YCB object models."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from volume_benchmark.common.geometry import invert_T, make_T
from volume_benchmark.common.io import Frame, save_prepared_scan
from volume_benchmark.common.mesh_volume import (
    compute_mesh_volume_m3,
    load_mesh_as_meters,
    write_gt_volume_json,
)


def ycb_pose_txt_to_T(pose_path: Path) -> np.ndarray:
    """
    Load a 4x4 pose matrix from YCB text format (meters, camera-to-model or model-in-camera).

    Expects 4 lines of space-separated floats.
    """
    rows = []
    with pose_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])
    T = np.array(rows, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose in {pose_path}, got shape {T.shape}")
    return T


def convert_ycb_frame(
    depth_path: Path,
    mask_path: Path,
    pose_path: Path,
    K: np.ndarray,
    pose_is_cam_to_object: bool = True,
    depth_scale: float = 0.001,
) -> Frame:
    """Convert one YCB frame. Depth PNGs are typically uint16 millimeters."""
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth: {depth_path}")
    depth_m = depth_raw.astype(np.float32) * depth_scale

    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    T = ycb_pose_txt_to_T(pose_path)
    T_cam_to_object = T if pose_is_cam_to_object else invert_T(T)

    return Frame(
        depth_m=depth_m,
        mask=mask_raw > 0,
        T_cam_to_object=T_cam_to_object,
        source_info={
            "dataset": "ycb",
            "depth_path": str(depth_path),
            "mask_path": str(mask_path),
            "pose_path": str(pose_path),
        },
    )


def prepare_ycb_scan(
    output_dir: str | Path,
    mesh_path: str | Path,
    K: np.ndarray,
    frames: list[tuple[Path, Path, Path]],
    mesh_units: str = "m",
    repair_mesh: bool = False,
    depth_scale: float = 0.001,
    pose_is_cam_to_object: bool = True,
    metadata: dict | None = None,
) -> Path:
    """Prepare a normalized scan from YCB-style frame triplets (depth, mask, pose)."""
    if not frames:
        raise ValueError("At least one frame is required")

    converted = [
        convert_ycb_frame(
            depth_path=d,
            mask_path=m,
            pose_path=p,
            K=K,
            pose_is_cam_to_object=pose_is_cam_to_object,
            depth_scale=depth_scale,
        )
        for d, m, p in frames
    ]

    mesh = load_mesh_as_meters(mesh_path, source_units=mesh_units)
    volume_m3, watertight, gt_type = compute_mesh_volume_m3(mesh, repair=repair_mesh)

    out = Path(output_dir).expanduser().resolve()
    meta = dict(metadata or {})
    meta.update({"dataset": "ycb", "num_frames": len(converted)})
    save_prepared_scan(out, K, converted, mesh_path, metadata=meta)

    write_gt_volume_json(
        out / "gt_volume.json",
        volume_m3=volume_m3,
        method=gt_type,
        watertight=watertight,
        source_mesh=mesh_path,
    )
    return out

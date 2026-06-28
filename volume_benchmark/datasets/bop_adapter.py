"""Adapter for BOP-style dataset layouts."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from volume_benchmark.common.geometry import make_T
from volume_benchmark.common.io import Frame, save_prepared_scan
from volume_benchmark.common.mesh_volume import (
    compute_mesh_volume_m3,
    load_mesh_as_meters,
    write_gt_volume_json,
)


def _load_bop_camera(json_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with json_path.open("r", encoding="utf-8") as f:
        cam = json.load(f)
    K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
    R = np.array(cam["cam_R_w2c"], dtype=np.float64).reshape(3, 3)
    t = np.array(cam["cam_t_w2c"], dtype=np.float64).reshape(3)
    # BOP stores translation in mm.
    t_m = t / 1000.0
    return K, R, t_m


def _bop_pose_to_T_cam_to_object(R_w2c: np.ndarray, t_w2c_m: np.ndarray) -> np.ndarray:
    """
    Convert BOP world-to-camera pose to T_cam_to_object.

    Assumes the object/model frame is aligned with the BOP model coordinate frame
    (origin at model center, meters after scaling).
    """
    T_cam_from_world = make_T(R_w2c, t_w2c_m)
    # For synthetic BOP scenes, object pose in world is identity when model is centered.
    # Caller should pass T_world_from_object when available.
    return T_cam_from_world


def convert_bop_scene_sample(
    depth_path: Path,
    mask_path: Path,
    camera_json_path: Path,
    T_world_from_object: np.ndarray | None = None,
    depth_scale: float = 0.001,
) -> tuple[np.ndarray, Frame]:
    """
    Convert one BOP RGB-D sample into intrinsics and a normalized Frame.

    depth_scale converts stored depth units to meters (BOP depth PNGs are typically mm).
    """
    K, R_w2c, t_w2c_m = _load_bop_camera(camera_json_path)

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    depth_m = depth_raw.astype(np.float32) * depth_scale

    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")
    mask = mask_raw > 0

    T_cam_from_world = make_T(R_w2c, t_w2c_m)
    if T_world_from_object is None:
        T_world_from_object = np.eye(4, dtype=np.float64)
    T_cam_to_object = T_cam_from_world @ T_world_from_object

    frame = Frame(
        depth_m=depth_m,
        mask=mask,
        T_cam_to_object=T_cam_to_object,
        source_info={
            "dataset": "bop",
            "depth_path": str(depth_path),
            "mask_path": str(mask_path),
            "camera_json_path": str(camera_json_path),
        },
    )
    return K, frame


def prepare_bop_scan(
    output_dir: str | Path,
    mesh_path: str | Path,
    frames: list[tuple[Path, Path, Path]],
    T_world_from_object: np.ndarray | None = None,
    mesh_units: str = "auto",
    repair_mesh: bool = False,
    depth_scale: float = 0.001,
    metadata: dict | None = None,
) -> Path:
    """
    Prepare a normalized scan from BOP-style inputs.

    frames: list of (depth_png, mask_png, scene_camera_json) per view.
    """
    if not frames:
        raise ValueError("At least one frame is required")

    converted_frames: list[Frame] = []
    K: np.ndarray | None = None
    for depth_path, mask_path, cam_path in frames:
        K_i, frame = convert_bop_scene_sample(
            depth_path=depth_path,
            mask_path=mask_path,
            camera_json_path=cam_path,
            T_world_from_object=T_world_from_object,
            depth_scale=depth_scale,
        )
        if K is None:
            K = K_i
        elif not np.allclose(K, K_i, rtol=1e-4, atol=1e-6):
            raise ValueError("All BOP frames must share the same intrinsics K")
        converted_frames.append(frame)

    mesh = load_mesh_as_meters(mesh_path, source_units=mesh_units)
    volume_m3, watertight, gt_type = compute_mesh_volume_m3(mesh, repair=repair_mesh)

    out = Path(output_dir).expanduser().resolve()
    meta = dict(metadata or {})
    meta.update({"dataset": "bop", "num_frames": len(converted_frames)})
    save_prepared_scan(out, K, converted_frames, mesh_path, metadata=meta)

    write_gt_volume_json(
        out / "gt_volume.json",
        volume_m3=volume_m3,
        method=gt_type,
        watertight=watertight,
        source_mesh=mesh_path,
    )
    return out

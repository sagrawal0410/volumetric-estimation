"""Build canonical manifest rows from ZED captures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from volrecon.config import PROJECT_ROOT
from volrecon.datasets.canonical_schema import StereoCalibration, ViewRecord
from volrecon.io.json_io import write_json, write_jsonl


INFERENCE_MODALITIES_ZED = ["left", "right", "left_rgb", "right_rgb"]
FORBIDDEN_MODALITIES_ZED = [
    "zed_depth",
    "zed_pointcloud",
    "zed_spatial_mapping",
    "zed_confidence",
]


def build_view_meta(
    frame_idx: int,
    timestamp_ns: int,
    paths: dict[str, str],
    calib: dict[str, Any],
    T_world_left: np.ndarray | None,
    tracking_state: str | None,
    camera_serial: str,
    sdk_version: str,
) -> dict[str, Any]:
    return {
        "frame_idx": frame_idx,
        "timestamp_ns": int(timestamp_ns),
        "left_path": paths["left"],
        "right_path": paths["right"],
        "K_left": np.asarray(calib["K_left"]).tolist(),
        "K_right": np.asarray(calib["K_right"]).tolist(),
        "T_left_right": np.asarray(calib["T_left_right"]).tolist(),
        "baseline_m": float(calib["baseline_m"]),
        "T_world_left": T_world_left.tolist() if T_world_left is not None else None,
        "tracking_state": tracking_state,
        "camera_serial": camera_serial,
        "sdk_version": sdk_version,
        "image_width": int(calib["image_width"]),
        "image_height": int(calib["image_height"]),
        "inference_allowed_modalities": INFERENCE_MODALITIES_ZED,
        "forbidden_modalities": FORBIDDEN_MODALITIES_ZED,
        "notes": ["ZED capture: rectified RGB only; depth from FoundationStereo."],
    }


def frame_to_view_record(
    scene_id: str,
    view_id: str,
    paths: dict[str, str],
    calib: dict[str, Any],
    meta: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
) -> ViewRecord:
    K_left = np.asarray(calib["K_left"], dtype=np.float64)
    T_wl = meta.get("T_world_left")
    T_world_cam = np.asarray(T_wl, dtype=np.float64).reshape(4, 4) if T_wl is not None else None
    T_cw = None
    if T_world_cam is not None:
        from volrecon.geometry.transforms import invert_T

        T_cw = invert_T(T_world_cam)

    left_p = paths["left"]
    stereo = StereoCalibration(
        has_true_stereo=True,
        baseline_m=float(calib["baseline_m"]),
        T_left_right=np.asarray(calib["T_left_right"], dtype=np.float64),
        rectified=True,
        source="zed_rectified",
    )

    rec = ViewRecord(
        dataset="zed_live",
        scene_id=scene_id,
        view_id=view_id,
        rgb_path=left_p,
        left_path=left_p,
        right_path=paths["right"],
        K=K_left,
        T_world_cam=T_world_cam,
        T_cam_world=T_cw,
        stereo=stereo,
        notes=meta.get("notes", []),
        sensor=f"ZED:{meta.get('camera_serial', 'unknown')}",
    )
    rec.inference_allowed_modalities = sorted(set(rec.inference_allowed_modalities) | {"left_rgb", "right_rgb"})
    return rec


def write_scene_manifest(scene_dir: Path, views: list[ViewRecord], project_root: Path = PROJECT_ROOT) -> Path:
    manifest_path = scene_dir / "manifest.jsonl"
    rows = [v.to_dict(project_root) for v in views]
    write_jsonl(manifest_path, rows)
    return manifest_path


def write_scene_meta(scene_dir: Path, scene_id: str, extra: dict[str, Any] | None = None) -> None:
    data = {"dataset": "zed_live", "scene_id": scene_id, "num_views": 0}
    if extra:
        data.update(extra)
    write_json(scene_dir / "scene_meta.json", data)

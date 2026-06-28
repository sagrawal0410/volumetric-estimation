"""Load normalized T-LESS prepared scans."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class PreparedFrame:
    index: int
    rgb: np.ndarray
    depth_m: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    T_cam_to_object: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedScan:
    scan_dir: Path
    frames: list[PreparedFrame]
    gt_volume: dict[str, Any]
    selected_views: dict[str, Any] | None = None


def load_prepared_scan(scan_dir: str | Path) -> PreparedScan:
    root = Path(scan_dir).expanduser().resolve()
    frames_dir = root / "frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Missing frames/ under {root}")

    gt_path = root / "gt_volume.json"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing gt_volume.json under {root}")
    with gt_path.open("r", encoding="utf-8") as f:
        gt_volume = json.load(f)

    selected_views = None
    sv_path = root / "selected_views.json"
    if sv_path.is_file():
        with sv_path.open("r", encoding="utf-8") as f:
            selected_views = json.load(f)

    frame_indices = sorted(
        int(p.name.split("_")[1])
        for p in frames_dir.glob("frame_*_depth.npy")
    )
    frames: list[PreparedFrame] = []
    for idx in frame_indices:
        prefix = f"frame_{idx:03d}"
        depth_m = np.load(frames_dir / f"{prefix}_depth.npy")
        mask = cv2.imread(str(frames_dir / f"{prefix}_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        K = np.load(frames_dir / f"{prefix}_K.npy")
        T = np.load(frames_dir / f"{prefix}_T_cam_to_object.npy")
        rgb_path = frames_dir / f"{prefix}_rgb.png"
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        meta: dict[str, Any] = {}
        meta_path = frames_dir / f"{prefix}_meta.json"
        if meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        frames.append(
            PreparedFrame(
                index=idx,
                rgb=rgb,
                depth_m=depth_m.astype(np.float32),
                mask=mask,
                K=K.astype(np.float64),
                T_cam_to_object=T.astype(np.float64),
                meta=meta,
            )
        )

    if not frames:
        raise ValueError(f"No frames found in {frames_dir}")

    return PreparedScan(
        scan_dir=root,
        frames=frames,
        gt_volume=gt_volume,
        selected_views=selected_views,
    )


def gt_comparison_fields(scan: PreparedScan, volume_m3: float | None) -> dict[str, Any]:
    gt_m3 = scan.gt_volume.get("volume_m3")
    gt_cm3 = scan.gt_volume.get("volume_cm3")
    if gt_m3 is None:
        return {
            "gt_volume_m3": None,
            "gt_volume_cm3": gt_cm3,
            "relative_error_percent": None,
        }
    if volume_m3 is None:
        return {
            "gt_volume_m3": gt_m3,
            "gt_volume_cm3": gt_cm3,
            "relative_error_percent": None,
        }
    rel = abs(volume_m3 - gt_m3) / gt_m3 * 100.0 if gt_m3 > 0 else None
    return {
        "gt_volume_m3": gt_m3,
        "gt_volume_cm3": gt_cm3,
        "relative_error_percent": rel,
    }


def resolve_output_dir(scan_dir: Path, method: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()
    return scan_dir / "outputs" / method


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

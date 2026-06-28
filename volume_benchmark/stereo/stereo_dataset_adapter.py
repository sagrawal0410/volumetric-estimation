"""Normalized prepared stereo scan I/O."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

STEREO_FRAME_PATTERN = re.compile(
    r"^frame_(\d{3})_(left\.png|right\.png|mask\.png|T_left_cam_to_object\.npy|meta\.json)$"
)


@dataclass
class StereoFrame:
    index: int
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    mask: np.ndarray
    T_left_cam_to_object: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedStereoScan:
    scan_dir: Path
    K_left: np.ndarray
    baseline_m: float
    frames: list[StereoFrame]
    gt_mesh_path: Path
    gt_volume: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_gt_volume(scan_dir: Path) -> dict[str, Any]:
    with (scan_dir / "gt_volume.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prepared_stereo_scan(scan_dir: str | Path) -> PreparedStereoScan:
    root = Path(scan_dir).expanduser().resolve()
    if not (root / "K_left.npy").is_file():
        raise FileNotFoundError(f"Missing K_left.npy in {root}")
    K_left = np.load(root / "K_left.npy")
    with (root / "baseline_m.json").open("r", encoding="utf-8") as f:
        baseline_m = float(json.load(f)["baseline_m"])

    gt_mesh = root / "gt_mesh.ply"
    if not gt_mesh.is_file():
        raise FileNotFoundError(f"Missing gt_mesh.ply in {root}")

    meta_path = root / "metadata.json"
    metadata = {}
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    frames_dir = root / "frames"
    indices = sorted(
        int(p.name.split("_")[1])
        for p in frames_dir.glob("frame_*_left.png")
    )
    frames: list[StereoFrame] = []
    for idx in indices:
        prefix = f"frame_{idx:03d}"
        left = cv2.cvtColor(cv2.imread(str(frames_dir / f"{prefix}_left.png")), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(cv2.imread(str(frames_dir / f"{prefix}_right.png")), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(frames_dir / f"{prefix}_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        T = np.load(frames_dir / f"{prefix}_T_left_cam_to_object.npy")
        fmeta = {}
        mp = frames_dir / f"{prefix}_meta.json"
        if mp.is_file():
            with mp.open("r", encoding="utf-8") as f:
                fmeta = json.load(f)
        frames.append(
            StereoFrame(index=idx, left_rgb=left, right_rgb=right, mask=mask, T_left_cam_to_object=T, meta=fmeta)
        )

    return PreparedStereoScan(
        scan_dir=root,
        K_left=K_left,
        baseline_m=baseline_m,
        frames=frames,
        gt_mesh_path=gt_mesh,
        gt_volume=_load_gt_volume(root),
        metadata=metadata,
    )


def save_prepared_stereo_scan(
    scan_dir: str | Path,
    K_left: np.ndarray,
    baseline_m: float,
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict | None]],
    gt_mesh_path: str | Path,
    gt_volume: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Save stereo scan. Each frame tuple: left_rgb, right_rgb, mask, T_left_cam_to_object, optional meta.
    """
    root = Path(scan_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "K_left.npy", np.asarray(K_left, dtype=np.float64))
    with (root / "baseline_m.json").open("w", encoding="utf-8") as f:
        json.dump({"baseline_m": float(baseline_m)}, f, indent=2)

    gt_src = Path(gt_mesh_path).resolve()
    gt_dst = root / "gt_mesh.ply"
    if gt_src != gt_dst:
        shutil.copy2(gt_src, gt_dst)
    with (root / "gt_volume.json").open("w", encoding="utf-8") as f:
        json.dump(gt_volume, f, indent=2)

    if metadata:
        with (root / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, (left, right, mask, T, fmeta) in enumerate(frames):
        prefix = f"frame_{idx:03d}"
        cv2.imwrite(str(frames_dir / f"{prefix}_left.png"), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frames_dir / f"{prefix}_right.png"), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frames_dir / f"{prefix}_mask.png"), (mask.astype(np.uint8) * 255))
        np.save(frames_dir / f"{prefix}_T_left_cam_to_object.npy", T.astype(np.float64))
        if fmeta:
            with (frames_dir / f"{prefix}_meta.json").open("w", encoding="utf-8") as f:
                json.dump(fmeta, f, indent=2)
    return root


def validate_prepared_stereo_scan(scan_dir: str | Path) -> list[str]:
    errors: list[str] = []
    root = Path(scan_dir)
    for req in ("K_left.npy", "baseline_m.json", "gt_mesh.ply", "gt_volume.json", "frames"):
        if not (root / req).exists():
            errors.append(f"Missing {req}")
    if errors:
        return errors
    try:
        load_prepared_stereo_scan(root)
    except Exception as exc:
        errors.append(str(exc))
    return errors

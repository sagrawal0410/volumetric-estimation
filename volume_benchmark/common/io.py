"""Load, save, and validate the normalized prepared-scan format."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

FRAME_PREFIX = "frame_"
FRAME_PATTERN = re.compile(r"^frame_(\d{3})_(depth\.npy|mask\.png|T_cam_to_object\.npy)$")


@dataclass
class Frame:
    """Single depth view with pose and optional provenance metadata."""

    depth_m: np.ndarray
    mask: np.ndarray
    T_cam_to_object: np.ndarray
    source_info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.depth_m.dtype != np.float32:
            self.depth_m = self.depth_m.astype(np.float32)
        if self.mask.dtype != bool:
            self.mask = self.mask.astype(bool)
        self.T_cam_to_object = np.asarray(self.T_cam_to_object, dtype=np.float64)
        if self.T_cam_to_object.shape != (4, 4):
            raise ValueError(
                f"T_cam_to_object must have shape (4, 4), got {self.T_cam_to_object.shape}"
            )


@dataclass
class PreparedScan:
    """Fully loaded prepared scan directory."""

    scan_dir: Path
    K: np.ndarray
    frames: list[Frame]
    gt_mesh_path: Path
    gt_volume: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def _scan_dir_path(scan_dir: str | Path) -> Path:
    path = Path(scan_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Scan directory does not exist: {path}")
    return path


def _load_gt_volume(scan_dir: Path) -> dict[str, Any]:
    gt_path = scan_dir / "gt_volume.json"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing gt_volume.json in {scan_dir}")
    with gt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "volume_m3" not in data:
        raise ValueError(f"gt_volume.json in {scan_dir} is missing 'volume_m3'")
    return data


def _discover_frame_indices(frames_dir: Path) -> list[int]:
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Missing frames/ directory in scan: {frames_dir.parent}")

    indices: set[int] = set()
    for path in frames_dir.iterdir():
        match = FRAME_PATTERN.match(path.name)
        if match:
            indices.add(int(match.group(1)))

    if not indices:
        raise ValueError(f"No frame files found in {frames_dir}")

    missing: list[str] = []
    for idx in sorted(indices):
        prefix = f"{FRAME_PREFIX}{idx:03d}"
        for suffix in ("depth.npy", "mask.png", "T_cam_to_object.npy"):
            if not (frames_dir / f"{prefix}_{suffix}").is_file():
                missing.append(f"{prefix}_{suffix}")
    if missing:
        raise FileNotFoundError(
            f"Incomplete frames in {frames_dir}. Missing: {', '.join(missing)}"
        )
    return sorted(indices)


def _load_frame(frames_dir: Path, index: int, source_info: dict | None = None) -> Frame:
    prefix = f"{FRAME_PREFIX}{index:03d}"
    depth_path = frames_dir / f"{prefix}_depth.npy"
    mask_path = frames_dir / f"{prefix}_mask.png"
    pose_path = frames_dir / f"{prefix}_T_cam_to_object.npy"

    depth_m = np.load(depth_path)
    if depth_m.ndim != 2:
        raise ValueError(f"{depth_path} must be a 2-D depth map, got shape {depth_m.shape}")

    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")
    mask = mask_raw > 0

    T_cam_to_object = np.load(pose_path)
    return Frame(
        depth_m=depth_m,
        mask=mask,
        T_cam_to_object=T_cam_to_object,
        source_info=dict(source_info or {}),
    )


def load_prepared_scan(scan_dir: str | Path) -> PreparedScan:
    """Load a prepared scan directory into memory."""
    path = _scan_dir_path(scan_dir)

    k_path = path / "K.npy"
    if not k_path.is_file():
        raise FileNotFoundError(f"Missing K.npy in {path}")
    K = np.load(k_path)
    if K.shape != (3, 3):
        raise ValueError(f"K.npy must have shape (3, 3), got {K.shape}")

    gt_mesh_path = path / "gt_mesh.ply"
    if not gt_mesh_path.is_file():
        raise FileNotFoundError(f"Missing gt_mesh.ply in {path}")

    metadata_path = path / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    frames_dir = path / "frames"
    indices = _discover_frame_indices(frames_dir)
    frame_source = metadata.get("frame_source_info", {})
    frames = [
        _load_frame(frames_dir, idx, frame_source.get(str(idx), frame_source.get(idx, {})))
        for idx in indices
    ]

    gt_volume = _load_gt_volume(path)
    return PreparedScan(
        scan_dir=path,
        K=K,
        frames=frames,
        gt_mesh_path=gt_mesh_path,
        gt_volume=gt_volume,
        metadata=metadata,
    )


def save_prepared_scan(
    scan_dir: str | Path,
    K: np.ndarray,
    frames: list[Frame],
    gt_mesh_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a prepared scan directory in the normalized format."""
    path = Path(scan_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)

    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must have shape (3, 3), got {K.shape}")

    if not frames:
        raise ValueError("At least one frame is required")

    np.save(path / "K.npy", K)

    gt_src = Path(gt_mesh_path).expanduser().resolve()
    gt_dst = path / "gt_mesh.ply"
    if not gt_src.is_file():
        raise FileNotFoundError(f"Ground-truth mesh not found: {gt_src}")
    if gt_src != gt_dst:
        shutil.copy2(gt_src, gt_dst)

    frames_dir = path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(metadata or {})
    frame_source: dict[str, dict] = {}
    for idx, frame in enumerate(frames):
        prefix = f"{FRAME_PREFIX}{idx:03d}"
        np.save(frames_dir / f"{prefix}_depth.npy", frame.depth_m.astype(np.float32))
        mask_u8 = (frame.mask.astype(np.uint8)) * 255
        if not cv2.imwrite(str(frames_dir / f"{prefix}_mask.png"), mask_u8):
            raise RuntimeError(f"Failed to write mask for frame {idx}")
        np.save(frames_dir / f"{prefix}_T_cam_to_object.npy", frame.T_cam_to_object)
        if frame.source_info:
            frame_source[str(idx)] = frame.source_info
    if frame_source:
        meta["frame_source_info"] = frame_source

    if meta:
        with (path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return path


def validate_prepared_scan(scan_dir: str | Path) -> list[str]:
    """Validate a prepared scan directory. Returns a list of error messages."""
    errors: list[str] = []
    path = Path(scan_dir).expanduser()

    if not path.is_dir():
        return [f"Scan directory does not exist: {path}"]

    k_path = path / "K.npy"
    if not k_path.is_file():
        errors.append("Missing K.npy")
    else:
        try:
            K = np.load(k_path)
            if K.shape != (3, 3):
                errors.append(f"K.npy must be (3, 3), got {K.shape}")
            if not np.all(np.isfinite(K)):
                errors.append("K.npy contains non-finite values")
        except Exception as exc:
            errors.append(f"Failed to load K.npy: {exc}")

    gt_mesh = path / "gt_mesh.ply"
    if not gt_mesh.is_file():
        errors.append("Missing gt_mesh.ply")

    gt_vol_path = path / "gt_volume.json"
    if not gt_vol_path.is_file():
        errors.append("Missing gt_volume.json")
    else:
        try:
            with gt_vol_path.open("r", encoding="utf-8") as f:
                gt = json.load(f)
            for key in ("volume_m3", "gt_type", "watertight", "source_mesh"):
                if key not in gt:
                    errors.append(f"gt_volume.json missing required key: {key}")
            if "volume_m3" in gt and gt["volume_m3"] <= 0:
                errors.append("gt_volume.json volume_m3 must be positive")
        except Exception as exc:
            errors.append(f"Failed to read gt_volume.json: {exc}")

    frames_dir = path / "frames"
    if not frames_dir.is_dir():
        errors.append("Missing frames/ directory")
        return errors

    try:
        indices = _discover_frame_indices(frames_dir)
    except (FileNotFoundError, ValueError) as exc:
        errors.append(str(exc))
        return errors

    if len(indices) < 1:
        errors.append("Scan must contain at least one frame")

    for idx in indices:
        prefix = f"{FRAME_PREFIX}{idx:03d}"
        depth_path = frames_dir / f"{prefix}_depth.npy"
        mask_path = frames_dir / f"{prefix}_mask.png"
        pose_path = frames_dir / f"{prefix}_T_cam_to_object.npy"

        depth: np.ndarray | None = None
        try:
            depth = np.load(depth_path)
            if depth.dtype != np.float32:
                errors.append(f"{prefix}: depth must be float32, got {depth.dtype}")
            if depth.ndim != 2:
                errors.append(f"{prefix}: depth must be 2-D")
            elif not np.all(np.isfinite(depth[depth > 0])):
                errors.append(f"{prefix}: depth contains non-finite positive values")
        except Exception as exc:
            errors.append(f"{prefix}: failed to load depth: {exc}")

        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            errors.append(f"{prefix}: could not read mask PNG")
        elif not np.isin(mask_raw, [0, 255]).all():
            errors.append(f"{prefix}: mask must contain only 0 and 255")

        try:
            T = np.load(pose_path)
            if T.shape != (4, 4):
                errors.append(f"{prefix}: T_cam_to_object must be (4, 4)")
            elif not np.all(np.isfinite(T)):
                errors.append(f"{prefix}: T_cam_to_object contains non-finite values")
        except Exception as exc:
            errors.append(f"{prefix}: failed to load pose: {exc}")

        if depth is not None and mask_raw is not None and depth.shape != mask_raw.shape:
            errors.append(
                f"{prefix}: depth shape {depth.shape} != mask shape {mask_raw.shape}"
            )

    return errors

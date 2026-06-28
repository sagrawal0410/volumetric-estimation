"""Canonical dataset schema dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from volrecon.config import EVAL_ONLY_MODALITIES, INFERENCE_MODALITIES, INTERNAL_UNITS


def _path_to_str(p: Path | str | None) -> str | None:
    if p is None:
        return None
    return str(p)


def _ensure_4x4(T: np.ndarray | None) -> np.ndarray | None:
    if T is None:
        return None
    return np.asarray(T, dtype=np.float64).reshape(4, 4)


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    K: np.ndarray
    distortion: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["K"] = self.K.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraIntrinsics":
        K = np.asarray(d["K"], dtype=np.float64)
        return cls(
            width=int(d["width"]),
            height=int(d["height"]),
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            K=K,
            distortion=d.get("distortion"),
        )


@dataclass
class StereoCalibration:
    has_true_stereo: bool
    left_K: CameraIntrinsics | None = None
    right_K: CameraIntrinsics | None = None
    baseline_m: float | None = None
    T_left_right: np.ndarray | None = None
    rectified: bool = False
    source: str = "unknown"
    synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_true_stereo": self.has_true_stereo,
            "left_K": self.left_K.to_dict() if self.left_K else None,
            "right_K": self.right_K.to_dict() if self.right_K else None,
            "baseline_m": self.baseline_m,
            "T_left_right": self.T_left_right.tolist() if self.T_left_right is not None else None,
            "rectified": self.rectified,
            "source": self.source,
            "synthetic": self.synthetic,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "StereoCalibration | None":
        if d is None:
            return None
        T = d.get("T_left_right")
        return cls(
            has_true_stereo=bool(d["has_true_stereo"]),
            left_K=CameraIntrinsics.from_dict(d["left_K"]) if d.get("left_K") else None,
            right_K=CameraIntrinsics.from_dict(d["right_K"]) if d.get("right_K") else None,
            baseline_m=d.get("baseline_m"),
            T_left_right=np.asarray(T, dtype=np.float64) if T is not None else None,
            rectified=bool(d.get("rectified", False)),
            source=str(d.get("source", "unknown")),
            synthetic=bool(d.get("synthetic", False)),
        )


@dataclass
class ObjectPoseRecord:
    obj_id: int
    instance_id: int
    T_model_cam: np.ndarray
    T_cam_model: np.ndarray
    model_path: Path | str
    visible_fraction: float | None = None
    bbox_visib: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "obj_id": self.obj_id,
            "instance_id": self.instance_id,
            "T_model_cam": self.T_model_cam.tolist(),
            "T_cam_model": self.T_cam_model.tolist(),
            "model_path": _path_to_str(self.model_path),
            "visible_fraction": self.visible_fraction,
            "bbox_visib": self.bbox_visib,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObjectPoseRecord":
        return cls(
            obj_id=int(d["obj_id"]),
            instance_id=int(d["instance_id"]),
            T_model_cam=_ensure_4x4(d["T_model_cam"]),
            T_cam_model=_ensure_4x4(d["T_cam_model"]),
            model_path=d["model_path"],
            visible_fraction=d.get("visible_fraction"),
            bbox_visib=d.get("bbox_visib"),
        )


@dataclass
class ViewRecord:
    dataset: Literal["robi", "bop_tless", "zed_live"]
    scene_id: str
    view_id: str
    rgb_path: Path | str | None = None
    left_path: Path | str | None = None
    right_path: Path | str | None = None
    mono_path: Path | str | None = None
    gt_depth_path: Path | str | None = None
    mask_paths: list[Path | str] = field(default_factory=list)
    K: np.ndarray | None = None
    T_world_cam: np.ndarray | None = None
    T_cam_world: np.ndarray | None = None
    stereo: StereoCalibration | None = None
    available_modalities: list[str] = field(default_factory=list)
    inference_allowed_modalities: list[str] = field(default_factory=list)
    eval_only_modalities: list[str] = field(default_factory=list)
    object_poses: list[ObjectPoseRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    split: str | None = None
    synthetic: bool = False
    sensor: str | None = None
    original_units: str = INTERNAL_UNITS
    unit_conversion_applied: bool = True

    def __post_init__(self) -> None:
        self._refresh_modality_lists()

    def _refresh_modality_lists(self) -> None:
        available: list[str] = []
        if self.rgb_path:
            available.append("rgb")
        if self.left_path:
            available.append("left")
        if self.right_path:
            available.append("right")
        if self.mono_path:
            available.append("mono")
        if self.gt_depth_path:
            available.append("gt_depth")
        if self.mask_paths:
            available.append("mask")
        self.available_modalities = available

        inference = [m for m in available if m in INFERENCE_MODALITIES]
        if self.stereo and self.stereo.has_true_stereo:
            if "left" in available and "right" in available:
                inference = sorted(set(inference) | {"left", "right"})
        self.inference_allowed_modalities = sorted(set(inference))

        eval_only = [m for m in available if m in EVAL_ONLY_MODALITIES]
        if self.gt_depth_path:
            eval_only.append("gt_depth")
        self.eval_only_modalities = sorted(set(eval_only))

        # Safety: GT depth must never be inference-allowed.
        self.inference_allowed_modalities = [
            m for m in self.inference_allowed_modalities if m != "gt_depth"
        ]

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:
        def rel(p: Path | str | None) -> str | None:
            if p is None:
                return None
            path = Path(p)
            if project_root and path.is_absolute():
                try:
                    return str(path.relative_to(project_root))
                except ValueError:
                    return str(path)
            return str(path)

        return {
            "dataset": self.dataset,
            "scene_id": self.scene_id,
            "view_id": self.view_id,
            "split": self.split,
            "synthetic": self.synthetic,
            "sensor": self.sensor,
            "rgb_path": rel(self.rgb_path),
            "left_path": rel(self.left_path),
            "right_path": rel(self.right_path),
            "mono_path": rel(self.mono_path),
            "gt_depth_path": rel(self.gt_depth_path),
            "mask_paths": [rel(p) for p in self.mask_paths],
            "K": self.K.tolist() if self.K is not None else None,
            "T_world_cam": self.T_world_cam.tolist() if self.T_world_cam is not None else None,
            "T_cam_world": self.T_cam_world.tolist() if self.T_cam_world is not None else None,
            "stereo": self.stereo.to_dict() if self.stereo else None,
            "available_modalities": self.available_modalities,
            "inference_allowed_modalities": self.inference_allowed_modalities,
            "eval_only_modalities": self.eval_only_modalities,
            "object_poses": [op.to_dict() for op in self.object_poses],
            "notes": self.notes,
            "original_units": self.original_units,
            "unit_conversion_applied": self.unit_conversion_applied,
            "units": INTERNAL_UNITS,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ViewRecord":
        K = d.get("K")
        T_wc = d.get("T_world_cam")
        T_cw = d.get("T_cam_world")
        rec = cls(
            dataset=d["dataset"],
            scene_id=str(d["scene_id"]),
            view_id=str(d["view_id"]),
            rgb_path=d.get("rgb_path"),
            left_path=d.get("left_path"),
            right_path=d.get("right_path"),
            mono_path=d.get("mono_path"),
            gt_depth_path=d.get("gt_depth_path"),
            mask_paths=d.get("mask_paths") or [],
            K=np.asarray(K, dtype=np.float64).reshape(3, 3) if K is not None else None,
            T_world_cam=_ensure_4x4(T_wc),
            T_cam_world=_ensure_4x4(T_cw),
            stereo=StereoCalibration.from_dict(d.get("stereo")),
            object_poses=[ObjectPoseRecord.from_dict(op) for op in d.get("object_poses", [])],
            notes=d.get("notes") or [],
            split=d.get("split"),
            synthetic=bool(d.get("synthetic", False)),
            sensor=d.get("sensor"),
            original_units=d.get("original_units", INTERNAL_UNITS),
            unit_conversion_applied=bool(d.get("unit_conversion_applied", True)),
        )
        rec._refresh_modality_lists()
        return rec


@dataclass
class SceneRecord:
    dataset: str
    scene_id: str
    views: list[ViewRecord] = field(default_factory=list)
    object_model_paths: dict[int, Path | str] = field(default_factory=dict)
    scene_gt_mesh_path: Path | str | None = None
    scene_gt_pointcloud_path: Path | str | None = None
    units: str = INTERNAL_UNITS
    original_units: str = INTERNAL_UNITS
    notes: list[str] = field(default_factory=list)
    split: str | None = None

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:
        def rel(p: Path | str | None) -> str | None:
            if p is None:
                return None
            path = Path(p)
            if project_root and path.is_absolute():
                try:
                    return str(path.relative_to(project_root))
                except ValueError:
                    return str(path)
            return str(path)

        return {
            "dataset": self.dataset,
            "scene_id": self.scene_id,
            "split": self.split,
            "views": [v.view_id for v in self.views],
            "num_views": len(self.views),
            "object_model_paths": {str(k): rel(v) for k, v in self.object_model_paths.items()},
            "scene_gt_mesh_path": rel(self.scene_gt_mesh_path),
            "scene_gt_pointcloud_path": rel(self.scene_gt_pointcloud_path),
            "units": self.units,
            "original_units": self.original_units,
            "notes": self.notes,
        }

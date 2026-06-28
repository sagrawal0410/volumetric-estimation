"""ROBI dataset adapter with recursive layout discovery."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from volrecon.config import PreprocessConfig
from volrecon.datasets.canonical_schema import (
    CameraIntrinsics,
    ObjectPoseRecord,
    SceneRecord,
    StereoCalibration,
    ViewRecord,
)
from volrecon.datasets.common import (
    classify_file,
    copy_or_symlink,
    detect_sensor,
    extract_view_id,
    recursive_scan,
    write_inspection_report,
)
from volrecon.datasets.preprocessing import (
    build_view_directory,
    save_placeholder_estimated_depth,
    write_scene_outputs,
)
from volrecon.geometry.transforms import invert_T, make_T
from volrecon.io.image_io import image_shape

logger = logging.getLogger(__name__)


def inspect_robi(root: Path, out_report: Path) -> dict:
    files = recursive_scan(root)
    file_counts = Counter(p.suffix.lower() for p in files)
    classified: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        for label in classify_file(p):
            classified[label].append(p)

    warnings: list[str] = []
    notes: list[str] = []

    left = classified.get("left", [])
    right = classified.get("right", [])
    if left and right:
        notes.append(f"Found {len(left)} left and {len(right)} right candidates — possible true stereo.")
    elif classified.get("rgb") and classified.get("depth"):
        warnings.append("RGB + depth detected but no left/right pairs — has_true_stereo will be false.")
    else:
        warnings.append("Could not confidently detect stereo pairs from filenames alone.")

    write_inspection_report(out_report, "ROBI", root, file_counts, classified, warnings, notes)
    return {"classified": classified, "warnings": warnings, "notes": notes}


def _parse_intrinsics_file(path: Path) -> tuple[np.ndarray, int, int] | None:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if "K" in data:
            K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
            w, h = int(data.get("width", 0)), int(data.get("height", 0))
            return K, w, h
        if all(k in data for k in ("fx", "fy", "cx", "cy")):
            K = np.array(
                [[data["fx"], 0, data["cx"]], [0, data["fy"], data["cy"]], [0, 0, 1]],
                dtype=np.float64,
            )
            w, h = int(data.get("width", 0)), int(data.get("height", 0))
            return K, w, h
    return None


def _parse_pose_file(path: Path) -> np.ndarray | None:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if "T_world_cam" in data:
            return np.asarray(data["T_world_cam"], dtype=np.float64).reshape(4, 4)
        if "R" in data and "t" in data:
            return make_T(np.asarray(data["R"]), np.asarray(data["t"]))
    if path.suffix.lower() == ".txt":
        vals = [float(x) for x in path.read_text().split()]
        if len(vals) == 16:
            return np.asarray(vals, dtype=np.float64).reshape(4, 4)
    return None


def _infer_scene_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    # Heuristic: scene = first 2-3 path components excluding filename
    if len(parts) >= 3:
        return "/".join(parts[:-2])
    if len(parts) >= 2:
        return parts[0]
    return "default_scene"


def _pair_stereo(left_paths: list[Path], right_paths: list[Path]) -> dict[str, tuple[Path | None, Path | None]]:
    left_by_id = {extract_view_id(p): p for p in left_paths}
    right_by_id = {extract_view_id(p): p for p in right_paths}
    view_ids = sorted(set(left_by_id) | set(right_by_id))
    return {vid: (left_by_id.get(vid), right_by_id.get(vid)) for vid in view_ids}


def _find_mono_pairs(mono_paths: list[Path]) -> dict[str, tuple[Path, Path] | None]:
    """Pair mono/gray images that look like stereo by shared prefix with _L/_R suffix."""
    pairs: dict[str, tuple[Path, Path]] = {}
    by_prefix: dict[str, dict[str, Path]] = defaultdict(dict)
    for p in mono_paths:
        stem = p.stem.lower()
        if stem.endswith("_l") or stem.endswith("-l"):
            by_prefix[stem[:-2]]["left"] = p
        elif stem.endswith("_r") or stem.endswith("-r"):
            by_prefix[stem[:-2]]["right"] = p
    for prefix, sides in by_prefix.items():
        if "left" in sides and "right" in sides:
            pairs[prefix] = (sides["left"], sides["right"])
    return pairs


def extract_robi(
    root: Path,
    out_dir: Path,
    manifest_path: Path,
    cfg: PreprocessConfig,
    report_path: Path | None = None,
) -> list[ViewRecord]:
    root = root.resolve()
    out_dir = out_dir.resolve()
    report_path = report_path or (cfg.project_root / "dataset_inspection_report_robi.md")
    inspection = inspect_robi(root, report_path)
    classified = inspection["classified"]

    left_paths = classified.get("left", [])
    right_paths = classified.get("right", [])
    rgb_paths = classified.get("rgb", [])
    mono_paths = classified.get("mono", [])
    gt_depth_paths = classified.get("gt_depth", []) or classified.get("depth", [])
    mask_paths_all = classified.get("mask", [])
    camera_files = classified.get("camera", [])
    pose_files = classified.get("pose", [])
    mesh_files = [p for p in recursive_scan(root) if p.suffix.lower() in {".ply", ".obj"}]

    has_true_stereo = bool(left_paths and right_paths)
    mono_stereo_pairs = _find_mono_pairs(mono_paths) if not has_true_stereo else {}

    # Group rgb/depth by scene heuristic
    scene_groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for p in rgb_paths:
        scene_groups[_infer_scene_key(p, root)]["rgb"].append(p)
    for p in gt_depth_paths:
        scene_groups[_infer_scene_key(p, root)]["gt_depth"].append(p)
    for p in mask_paths_all:
        scene_groups[_infer_scene_key(p, root)]["mask"].append(p)

    if not scene_groups:
        # Fall back: one scene per top-level folder
        for child in sorted(root.iterdir()):
            if child.is_dir():
                scene_groups[child.name] = {"rgb": [p for p in recursive_scan(child) if "rgb" in classify_file(p)]}

    all_views: list[ViewRecord] = []
    if manifest_path.exists() and cfg.overwrite:
        manifest_path.unlink()

    for scene_id, groups in sorted(scene_groups.items()):
        scene_root = out_dir / scene_id.replace("/", "_")
        gt_dir = scene_root / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        object_model_paths: dict[int, Path] = {}
        scene_notes: list[str] = []

        # Copy/symlink object models
        for i, mesh_path in enumerate(mesh_files):
            if scene_id.replace("_", "/") not in str(mesh_path) and scene_id not in str(mesh_path):
                continue
            dst = gt_dir / "object_models" / mesh_path.name
            copy_or_symlink(mesh_path, dst, cfg.symlink, cfg.overwrite)
            object_model_paths[i + 1] = dst

        scene_gt_mesh = None
        for mesh_path in mesh_files:
            if "scene" in mesh_path.stem.lower() or "gt" in mesh_path.stem.lower():
                dst = gt_dir / "scene_mesh_gt.ply"
                copy_or_symlink(mesh_path, dst, cfg.symlink, cfg.overwrite)
                scene_gt_mesh = dst
                break

        if scene_gt_mesh is None and groups.get("gt_depth") and pose_files:
            scene_notes.append(
                "GT scene mesh not found; GT depth + poses available — use fuse_gt_depth_to_mesh stub later."
            )

        views: list[ViewRecord] = []

        if has_true_stereo:
            pairs = _pair_stereo(left_paths, right_paths)
            for view_id, (lp, rp) in pairs.items():
                if lp is None or rp is None:
                    continue
                view_dir = build_view_directory(scene_root, view_id)
                left_dst = copy_or_symlink(lp, view_dir / "left.png", cfg.symlink, cfg.overwrite)
                right_dst = copy_or_symlink(rp, view_dir / "right.png", cfg.symlink, cfg.overwrite)
                w, h = image_shape(left_dst)
                K = np.eye(3, dtype=np.float64)
                K[0, 0] = K[1, 1] = max(w, h)
                K[0, 2], K[1, 2] = w / 2, h / 2
                for cf in camera_files:
                    parsed = _parse_intrinsics_file(cf)
                    if parsed:
                        K, w, h = parsed
                        break
                T_world_cam = None
                for pf in pose_files:
                    if view_id in pf.stem:
                        T_world_cam = _parse_pose_file(pf)
                        break
                if T_world_cam is None:
                    scene_notes.append(f"View {view_id}: no camera pose file found; T_world_cam=null.")

                gt_depth_dst = None
                for dp in groups.get("gt_depth", []):
                    if extract_view_id(dp) == view_id:
                        gt_depth_dst = copy_or_symlink(dp, view_dir / "gt_depth.png", cfg.symlink, cfg.overwrite)
                        break

                mask_dst: list[Path] = []
                for mp in groups.get("mask", []):
                    if extract_view_id(mp) == view_id:
                        mask_dst.append(copy_or_symlink(mp, view_dir / "mask.png", cfg.symlink, cfg.overwrite))

                save_placeholder_estimated_depth(view_dir)
                stereo = StereoCalibration(
                    has_true_stereo=True,
                    left_K=CameraIntrinsics(
                        width=w,
                        height=h,
                        fx=float(K[0, 0]),
                        fy=float(K[1, 1]),
                        cx=float(K[0, 2]),
                        cy=float(K[1, 2]),
                        K=K,
                    ),
                    right_K=None,
                    baseline_m=None,
                    rectified=False,
                    source="robi_filename_pair",
                )

                meta = {
                    "view_id": view_id,
                    "sensor": detect_sensor(str(lp)),
                    "original_units": "m",
                }
                (view_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

                view = ViewRecord(
                    dataset="robi",
                    scene_id=scene_id.replace("/", "_"),
                    view_id=view_id,
                    left_path=left_dst,
                    right_path=right_dst,
                    gt_depth_path=gt_depth_dst,
                    mask_paths=mask_dst,
                    K=K,
                    T_world_cam=T_world_cam,
                    T_cam_world=invert_T(T_world_cam) if T_world_cam is not None else None,
                    stereo=stereo,
                    sensor=detect_sensor(str(lp)),
                    notes=["True stereo pair detected from left/right filenames."],
                )
                views.append(view)
        elif mono_stereo_pairs:
            for view_id, (lp, rp) in mono_stereo_pairs.items():
                view_dir = build_view_directory(scene_root, view_id)
                left_dst = copy_or_symlink(lp, view_dir / "left.png", cfg.symlink, cfg.overwrite)
                right_dst = copy_or_symlink(rp, view_dir / "right.png", cfg.symlink, cfg.overwrite)
                w, h = image_shape(left_dst)
                K = np.eye(3)
                K[0, 0] = K[1, 1] = max(w, h)
                K[0, 2], K[1, 2] = w / 2, h / 2
                save_placeholder_estimated_depth(view_dir)
                stereo = StereoCalibration(
                    has_true_stereo=True,
                    left_K=CameraIntrinsics(width=w, height=h, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], K=K),
                    source="robi_mono_stereo",
                )
                view = ViewRecord(
                    dataset="robi",
                    scene_id=scene_id.replace("/", "_"),
                    view_id=view_id,
                    left_path=left_dst,
                    right_path=right_dst,
                    mono_path=left_dst,
                    K=K,
                    stereo=stereo,
                    sensor=detect_sensor(str(lp)),
                    notes=["Monochrome stereo pair accepted as mono_stereo modality."],
                )
                view.inference_allowed_modalities = sorted(set(view.inference_allowed_modalities) | {"mono_stereo"})
                views.append(view)
        else:
            for rgb_path in groups.get("rgb", []):
                view_id = extract_view_id(rgb_path)
                view_dir = build_view_directory(scene_root, view_id)
                rgb_dst = copy_or_symlink(rgb_path, view_dir / "rgb.png", cfg.symlink, cfg.overwrite)
                w, h = image_shape(rgb_dst)
                K = np.eye(3)
                K[0, 0] = K[1, 1] = max(w, h)
                K[0, 2], K[1, 2] = w / 2, h / 2
                gt_depth_dst = None
                for dp in groups.get("gt_depth", []):
                    if extract_view_id(dp) == view_id:
                        gt_depth_dst = copy_or_symlink(dp, view_dir / "gt_depth.png", cfg.symlink, cfg.overwrite)
                save_placeholder_estimated_depth(view_dir)
                stereo = StereoCalibration(has_true_stereo=False, source="robi_rgb_only")
                view = ViewRecord(
                    dataset="robi",
                    scene_id=scene_id.replace("/", "_"),
                    view_id=view_id,
                    rgb_path=rgb_dst,
                    gt_depth_path=gt_depth_dst,
                    K=K,
                    stereo=stereo,
                    sensor=detect_sensor(str(rgb_path)),
                    notes=["No true stereo pairs; RGB-only view."],
                )
                views.append(view)

        scene = SceneRecord(
            dataset="robi",
            scene_id=scene_id.replace("/", "_"),
            views=views,
            object_model_paths=object_model_paths,
            scene_gt_mesh_path=scene_gt_mesh,
            original_units="m",
            notes=scene_notes,
        )
        write_scene_outputs(scene, cfg, manifest_path)
        all_views.extend(views)

    logger.info("Extracted %d ROBI views across %d scenes", len(all_views), len(scene_groups))
    return all_views


def fuse_gt_depth_to_mesh_stub(scene_dir: Path) -> None:
    """Placeholder for future GT depth fusion into scene_mesh_gt.ply."""
    readme = scene_dir / "gt" / "FUSE_GT_DEPTH_README.txt"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "GT depth fusion stub: implement TSDF/poisson fusion from eval-only gt_depth frames.\n",
        encoding="utf-8",
    )

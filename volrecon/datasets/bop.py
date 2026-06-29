"""BOP / T-LESS dataset adapter."""

from __future__ import annotations

import json
import logging
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
    recursive_scan,
    write_inspection_report,
)
from volrecon.datasets.preprocessing import (
    build_view_directory,
    save_placeholder_estimated_depth,
    write_scene_outputs,
)
from volrecon.geometry.mesh_volume import voxelize_mesh
from volrecon.geometry.render_gt import (
    load_and_transform_bop_model,
    render_scene_gt_depth_from_objects,
    render_synthetic_stereo_pair,
)
from volrecon.geometry.transforms import bop_T_cam_model_to_meters, bop_T_model_cam_to_meters, invert_T, make_T
from volrecon.io.image_io import image_shape, read_image, write_image
from volrecon.io.json_io import read_json
from volrecon.io.mesh_io import load_mesh, save_mesh_ply

logger = logging.getLogger(__name__)

BOP_STEREO_WARNING = (
    "This split does not provide true stereo pairs; FoundationStereo inference cannot run "
    "directly without external stereo pairs or synthetic stereo rendering."
)


def inspect_bop(root: Path, split: str, out_report: Path) -> dict:
    split_dir = root / split if (root / split).exists() else root
    files = recursive_scan(split_dir)
    file_counts = Counter(p.suffix.lower() for p in files)
    classified: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        for label in classify_file(p):
            classified[label].append(p)

    warnings: list[str] = []
    notes: list[str] = []
    if not classified.get("left") and not classified.get("right"):
        warnings.append(BOP_STEREO_WARNING)
        notes.append("Standard BOP layout: rgb/gray + depth, no rectified left/right pairs.")

    write_inspection_report(out_report, f"BOP T-LESS ({split})", split_dir, file_counts, classified, warnings, notes)
    return {"classified": classified, "warnings": warnings, "split_dir": split_dir}


def _load_models(root: Path) -> tuple[dict[int, Path], dict]:
    models_dir = root / "models"
    if not models_dir.exists():
        models_dir = root / "models_eval"
    models_info_path = root / "models_info.json"
    models_info = read_json(models_info_path) if models_info_path.exists() else {}
    model_paths: dict[int, Path] = {}
    for p in sorted(models_dir.glob("obj_*.ply")):
        obj_id = int(p.stem.split("_")[1])
        model_paths[obj_id] = p
    return model_paths, models_info


def _bop_cam_pose_meters(cam_entry: dict) -> np.ndarray | None:
    if "cam_R_w2c" not in cam_entry or "cam_t_w2c" not in cam_entry:
        return None
    R = np.asarray(cam_entry["cam_R_w2c"], dtype=np.float64).reshape(3, 3)
    t_mm = np.asarray(cam_entry["cam_t_w2c"], dtype=np.float64).reshape(3)
    t_m = t_mm * 0.001
    return make_T(R, t_m)


def _parse_bop_scene(scene_dir: Path) -> tuple[dict, dict, dict | None]:
    scene_camera = read_json(scene_dir / "scene_camera.json")
    scene_gt = read_json(scene_dir / "scene_gt.json")
    gt_info_path = scene_dir / "scene_gt_info.json"
    scene_gt_info = read_json(gt_info_path) if gt_info_path.exists() else None
    return scene_camera, scene_gt, scene_gt_info


def _rgb_or_gray_path(scene_dir: Path, im_id: int) -> Path | None:
    for sub in ("rgb", "gray"):
        d = scene_dir / sub
        if not d.exists():
            continue
        for ext in (".png", ".jpg", ".tif"):
            p = d / f"{im_id:06d}{ext}"
            if p.exists():
                return p
    return None


def _depth_path(scene_dir: Path, im_id: int) -> Path | None:
    d = scene_dir / "depth"
    if not d.exists():
        return None
    for ext in (".png", ".tif"):
        p = d / f"{im_id:06d}{ext}"
        if p.exists():
            return p
    return None


def _mask_visib_paths(scene_dir: Path, im_id: int) -> list[Path]:
    d = scene_dir / "mask_visib"
    if not d.exists():
        return []
    return sorted(d.glob(f"{im_id:06d}_*.png"))


def extract_bop_tless(
    root: Path,
    split: str,
    out_dir: Path,
    manifest_path: Path,
    cfg: PreprocessConfig,
    mode: str = "real_rgb_only",
    baseline_m: float | None = None,
    report_path: Path | None = None,
    max_scenes: int | None = None,
    frame_stride: int = 1,
    max_views_per_scene: int | None = None,
    skip_union_voxels: bool = False,
) -> list[ViewRecord]:
    if mode not in {"real_rgb_only", "synthetic_stereo_from_bop_mesh"}:
        raise ValueError(f"Unknown mode: {mode}")

    root = root.resolve()
    out_dir = out_dir.resolve()
    baseline_m = baseline_m if baseline_m is not None else cfg.synthetic_baseline_m
    report_path = report_path or (cfg.project_root / "dataset_inspection_report_bop_tless.md")
    inspection = inspect_bop(root, split, report_path)
    split_dir: Path = inspection["split_dir"]
    model_paths, models_info = _load_models(root)

    if manifest_path.exists() and cfg.overwrite:
        manifest_path.unlink()

    all_views: list[ViewRecord] = []
    scene_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir() and (p / "scene_camera.json").exists()])
    if max_scenes is not None:
        scene_dirs = scene_dirs[:max_scenes]
        logger.info("Limiting to first %d BOP scenes", len(scene_dirs))

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        scene_camera, scene_gt, scene_gt_info = _parse_bop_scene(scene_dir)
        scene_out = out_dir / scene_id
        gt_dir = scene_out / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)

        # Copy object models (mm -> m on load later)
        object_model_paths: dict[int, Path] = {}
        obj_models_dir = gt_dir / "object_models"
        obj_models_dir.mkdir(parents=True, exist_ok=True)
        for obj_id, src in model_paths.items():
            dst = copy_or_symlink(src, obj_models_dir / src.name, cfg.symlink, cfg.overwrite)
            object_model_paths[obj_id] = dst

        scene_notes: list[str] = []
        has_world_pose = any(
            "cam_R_w2c" in scene_camera[str(i)] and "cam_t_w2c" in scene_camera[str(i)]
            for i in scene_camera
        )
        if not has_world_pose:
            scene_notes.append(
                "No world camera poses in scene_camera; object-centric frame used per instance."
            )

        views: list[ViewRecord] = []
        rendered_gt_dir = gt_dir / "rendered_gt_depth"
        rendered_gt_dir.mkdir(parents=True, exist_ok=True)
        transformed_mesh_dir = gt_dir / "object_meshes_in_scene_frame"
        transformed_mesh_dir.mkdir(parents=True, exist_ok=True)
        voxel_grids: list[np.ndarray] = []

        cam_items = sorted(scene_camera.items(), key=lambda x: int(x[0]))
        if frame_stride > 1:
            cam_items = cam_items[::frame_stride]
        if max_views_per_scene is not None:
            cam_items = cam_items[:max_views_per_scene]

        for im_id_str, cam_entry in cam_items:
            im_id = int(im_id_str)
            view_id = f"{im_id:06d}"
            view_dir = build_view_directory(scene_out, view_id)

            rgb_src = _rgb_or_gray_path(scene_dir, im_id)
            if rgb_src is None:
                continue
            rgb_dst = copy_or_symlink(rgb_src, view_dir / "rgb.png", cfg.symlink, cfg.overwrite)
            w, h = image_shape(rgb_dst)

            K = np.asarray(cam_entry["cam_K"], dtype=np.float64).reshape(3, 3)
            depth_scale = float(cam_entry.get("depth_scale", 1.0))
            T_world_cam = _bop_cam_pose_meters(cam_entry)
            T_cam_world = invert_T(T_world_cam) if T_world_cam is not None else None

            depth_src = _depth_path(scene_dir, im_id)
            gt_depth_dst = None
            if depth_src is not None:
                gt_depth_dst = copy_or_symlink(depth_src, view_dir / "gt_depth.png", cfg.symlink, cfg.overwrite)

            mask_dsts: list[Path] = []
            mask_visib_dir = view_dir / "mask_visib"
            mask_visib_dir.mkdir(exist_ok=True)
            for mp in _mask_visib_paths(scene_dir, im_id):
                mask_dsts.append(copy_or_symlink(mp, mask_visib_dir / mp.name, cfg.symlink, cfg.overwrite))

            save_placeholder_estimated_depth(view_dir)

            gt_instances = scene_gt.get(im_id_str, [])
            gt_info_instances = (scene_gt_info or {}).get(im_id_str, [])
            object_poses: list[ObjectPoseRecord] = []
            obj_meshes_cam: dict[int, np.ndarray] = {}
            mesh_cache: dict[int, object] = {}

            for idx, inst in enumerate(gt_instances):
                obj_id = int(inst["obj_id"])
                T_model_cam = bop_T_model_cam_to_meters(inst["cam_R_m2c"], inst["cam_t_m2c"])
                T_cam_model = bop_T_cam_model_to_meters(inst["cam_R_m2c"], inst["cam_t_m2c"])
                info = gt_info_instances[idx] if idx < len(gt_info_instances) else {}
                model_path = object_model_paths.get(obj_id)
                if model_path is None:
                    continue
                object_poses.append(
                    ObjectPoseRecord(
                        obj_id=obj_id,
                        instance_id=idx,
                        T_model_cam=T_model_cam,
                        T_cam_model=T_cam_model,
                        model_path=model_path,
                        visible_fraction=info.get("visib_fract"),
                        bbox_visib=info.get("bbox_visib"),
                    )
                )
                obj_meshes_cam[obj_id] = T_cam_model
                if obj_id not in mesh_cache:
                    mesh_cache[obj_id] = load_mesh(model_path)
                    mesh_cache[obj_id].apply_scale(0.001)

            # Render GT depth from meshes
            rendered_depth = render_scene_gt_depth_from_objects(
                mesh_cache,
                {k: obj_meshes_cam[k] for k in mesh_cache},
                K,
                w,
                h,
            )
            rendered_depth_path = rendered_gt_dir / f"{view_id}.npy"
            np.save(rendered_depth_path, rendered_depth.astype(np.float32))

            meta = {
                "view_id": view_id,
                "depth_scale": depth_scale,
                "depth_scale_note": "BOP depth PNG values * depth_scale = meters",
                "original_units": "mm",
                "unit_conversion_applied": True,
                "rendered_gt_depth_path": str(rendered_depth_path.relative_to(scene_out)),
            }
            (view_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            notes = [BOP_STEREO_WARNING] if mode == "real_rgb_only" else []
            stereo = StereoCalibration(
                has_true_stereo=False,
                source="bop_standard",
                synthetic=(mode == "synthetic_stereo_from_bop_mesh"),
            )

            left_path = right_path = None
            synthetic_flag = False

            if mode == "synthetic_stereo_from_bop_mesh":
                meshes = list(mesh_cache.values())
                poses = [obj_meshes_cam[oid] for oid in mesh_cache]
                if meshes:
                    synth = render_synthetic_stereo_pair(meshes, poses, K, w, h, baseline_m)
                    left_path = view_dir / "left.png"
                    right_path = view_dir / "right.png"
                    write_image(left_path, synth["left_rgb"])
                    write_image(right_path, synth["right_rgb"])
                    np.save(view_dir / "synthetic_left_depth_m.npy", synth["left_depth_m"])
                    stereo = StereoCalibration(
                        has_true_stereo=True,
                        left_K=CameraIntrinsics(
                            width=w, height=h, fx=float(K[0, 0]), fy=float(K[1, 1]),
                            cx=float(K[0, 2]), cy=float(K[1, 2]), K=K,
                        ),
                        right_K=CameraIntrinsics(
                            width=w, height=h, fx=float(K[0, 0]), fy=float(K[1, 1]),
                            cx=float(K[0, 2]), cy=float(K[1, 2]), K=K,
                        ),
                        baseline_m=baseline_m,
                        T_left_right=synth["T_left_right"],
                        rectified=True,
                        source="synthetic_from_bop_mesh",
                        synthetic=True,
                    )
                    synthetic_flag = True
                    notes.append("Synthetic stereo rendered from GT meshes; not for real benchmark scoring.")

            if len(gt_instances) > 1 and not has_world_pose:
                notes.append(
                    "Cluttered scene without world camera pose: scene-level multi-view fusion underdetermined."
                )

            view = ViewRecord(
                dataset="bop_tless",
                scene_id=scene_id,
                view_id=view_id,
                rgb_path=rgb_dst,
                left_path=left_path,
                right_path=right_path,
                gt_depth_path=gt_depth_dst,
                mask_paths=mask_dsts,
                K=K,
                T_world_cam=T_world_cam,
                T_cam_world=T_cam_world,
                stereo=stereo,
                object_poses=object_poses,
                notes=notes,
                split=split,
                synthetic=synthetic_flag,
                original_units="mm",
                unit_conversion_applied=True,
            )
            views.append(view)

            # Transformed meshes in canonical frame (model frame as world when no cam world pose)
            for op in object_poses:
                mesh = mesh_cache[op.obj_id]
                T_scene_model = invert_T(op.T_model_cam) if T_world_cam is None else invert_T(T_world_cam) @ invert_T(op.T_model_cam)
                transformed = load_and_transform_bop_model(
                    Path(op.model_path), T_scene_model, mm_to_m=False
                )
                out_mesh = transformed_mesh_dir / f"obj_{op.obj_id:06d}_inst{op.instance_id}.ply"
                save_mesh_ply(out_mesh, transformed)
                try:
                    if not skip_union_voxels:
                        voxel_grids.append(voxelize_mesh(transformed, cfg.voxel_size_m))
                except Exception:  # noqa: BLE001
                    pass

        if voxel_grids and not skip_union_voxels:
            union = voxel_grids[0].copy()
            for g in voxel_grids[1:]:
                if g.shape == union.shape:
                    union |= g
            np.savez_compressed(
                gt_dir / "union_gt_voxels.npz",
                voxels=union,
                voxel_size_m=cfg.voxel_size_m,
            )

        scene = SceneRecord(
            dataset="bop_tless",
            scene_id=scene_id,
            views=views,
            object_model_paths=object_model_paths,
            original_units="mm",
            notes=scene_notes,
            split=split,
        )
        write_scene_outputs(scene, cfg, manifest_path)
        all_views.extend(views)

    logger.info("Extracted %d BOP views from split %s (mode=%s)", len(all_views), split, mode)
    return all_views

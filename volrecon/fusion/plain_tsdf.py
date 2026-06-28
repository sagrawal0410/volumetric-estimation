"""High-level plain TSDF scene reconstruction from manifest + depth predictions."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from volrecon.config import PROJECT_ROOT
from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.fusion.bounds import (
    compute_bounds_from_depth_points,
    robust_expand_bounds,
    save_bounds_json,
)
from volrecon.fusion.fusion_utils import PoseConventionError, world_cam_from_view
from volrecon.fusion.open3d_tsdf import PlainTSDFConfig, PlainTSDFReconstructor
from volrecon.io.json_io import read_json, read_jsonl

logger = logging.getLogger(__name__)


@dataclass
class PlainTSDFRunConfig:
    manifest_path: Path
    depth_predictions_root: Path
    out_root: Path
    tsdf: PlainTSDFConfig
    project_root: Path = PROJECT_ROOT
    use_gt_bounds_for_debug: bool = False
    object_centric_bop: bool = True
    bounds_margin_m: float = 0.05


def _resolve(path: Path | str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root / p).resolve()


def group_views_by_scene(manifest_path: Path) -> dict[str, list[ViewRecord]]:
    records = [ViewRecord.from_dict(r) for r in read_jsonl(manifest_path)]
    groups: dict[str, list[ViewRecord]] = defaultdict(list)
    for r in records:
        groups[r.scene_id].append(r)
    return dict(groups)


def depth_prediction_dir(root: Path, scene_id: str, view_id: str) -> Path:
    return root / scene_id / view_id


def reconstruct_scene(
    scene_id: str,
    views: list[ViewRecord],
    cfg: PlainTSDFRunConfig,
) -> Path:
    views = sorted(views, key=lambda v: v.view_id)
    views = views[:: cfg.tsdf.frame_stride]
    if cfg.tsdf.max_views:
        views = views[: cfg.tsdf.max_views]

    depth_maps: list[np.ndarray] = []
    Ks: list[np.ndarray] = []
    T_world_cams: list[np.ndarray] = []
    frame_data: list[tuple[ViewRecord, Path, np.ndarray, np.ndarray]] = []

    object_centric = cfg.object_centric_bop and views[0].dataset == "bop_tless"

    for view in views:
        pred_dir = depth_prediction_dir(cfg.depth_predictions_root, scene_id, view.view_id)
        depth_path = pred_dir / "depth_m.npy"
        if not depth_path.exists():
            logger.warning("Skipping view %s: no depth prediction at %s", view.view_id, depth_path)
            continue

        K_path = pred_dir / "K_scaled.json"
        if K_path.exists():
            K = np.asarray(read_json(K_path)["K"], dtype=np.float64)
        elif view.K is not None:
            K = np.asarray(view.K, dtype=np.float64)
        else:
            logger.warning("Skipping view %s: no intrinsics", view.view_id)
            continue

        try:
            T_wc, T_cw = world_cam_from_view(view, object_centric=object_centric)
        except PoseConventionError as exc:
            if view.dataset == "robi":
                raise PoseConventionError(
                    f"ROBI scene {scene_id}: scene-level fusion requires camera poses. {exc}"
                ) from exc
            logger.warning("Skipping view %s: %s", view.view_id, exc)
            continue

        depth_m = np.load(depth_path).astype(np.float64)
        depth_maps.append(depth_m)
        Ks.append(K)
        T_world_cams.append(T_wc)
        frame_data.append((view, depth_path, K, T_cw))

    if not frame_data:
        raise PoseConventionError(f"No integrable views for scene {scene_id}")

    bounds = compute_bounds_from_depth_points(depth_maps, Ks, T_world_cams)
    bounds = robust_expand_bounds(bounds, cfg.bounds_margin_m)

    if cfg.use_gt_bounds_for_debug:
        scene_meta_path = cfg.project_root / "data" / "processed" / views[0].dataset / scene_id / "scene_meta.json"
        if scene_meta_path.exists():
            meta = read_json(scene_meta_path)
            gt_mesh = meta.get("scene_gt_mesh_path")
            if gt_mesh:
                import trimesh

                from volrecon.fusion.bounds import compute_bounds_from_gt_mesh

                mesh = trimesh.load(_resolve(gt_mesh, cfg.project_root), force="mesh")
                bounds = compute_bounds_from_gt_mesh(mesh)
                bounds = robust_expand_bounds(bounds, cfg.bounds_margin_m)

    out_dir = cfg.out_root / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)
    save_bounds_json(out_dir / "bounds.json", bounds, source="depth_pointcloud")

    recon = PlainTSDFReconstructor(cfg.tsdf, bounds)
    for view, depth_path, K, T_cw in frame_data:
        rgb_path = None
        if view.left_path:
            rgb_path = _resolve(view.left_path, cfg.project_root)
        elif view.rgb_path:
            rgb_path = _resolve(view.rgb_path, cfg.project_root)
        recon.integrate_view(
            rgb_path=rgb_path,
            depth_m_path=depth_path,
            K=K,
            T_cam_world=T_cw,
            view_id=view.view_id,
        )

    recon.save_outputs(out_dir)
    logger.info("Saved TSDF reconstruction for scene %s to %s", scene_id, out_dir)
    return out_dir


def run_plain_tsdf(cfg: PlainTSDFRunConfig) -> list[Path]:
    scenes = group_views_by_scene(cfg.manifest_path)
    outputs: list[Path] = []
    for scene_id, views in sorted(scenes.items()):
        try:
            outputs.append(reconstruct_scene(scene_id, views, cfg))
        except PoseConventionError as exc:
            logger.error("Scene %s skipped: %s", scene_id, exc)
    return outputs

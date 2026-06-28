"""High-level uncertainty map orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from volrecon.config import PROJECT_ROOT
from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.fusion.fusion_utils import world_cam_from_view
from volrecon.io.image_io import read_rgb
from volrecon.io.json_io import read_json, write_json
from volrecon.stereo.foundation_stereo_wrapper import resolve_view_paths
from volrecon.uncertainty.calibration import UncertaintyConfig
from volrecon.uncertainty.confidence_sources import compute_confidence_maps, save_confidence_maps
from volrecon.uncertainty.multiview_consistency import multiview_agreement_confidence, select_neighbor_views

logger = logging.getLogger(__name__)


def load_K_from_prediction(pred_dir: Path, view: ViewRecord) -> np.ndarray:
    k_path = pred_dir / "K_scaled.json"
    if k_path.exists():
        return np.asarray(read_json(k_path)["K"], dtype=np.float64)
    return np.asarray(view.K, dtype=np.float64)


def compute_view_uncertainty(
    view: ViewRecord,
    depth_pred_dir: Path,
    out_dir: Path,
    cfg: UncertaintyConfig,
    scene_views: list[ViewRecord] | None = None,
    depth_pred_root: Path | None = None,
    project_root: Path = PROJECT_ROOT,
    disparity_r2l: np.ndarray | None = None,
) -> None:
    depth_m = np.load(depth_pred_dir / "depth_m.npy").astype(np.float64)
    disparity = np.load(depth_pred_dir / "disparity.npy").astype(np.float64)
    left, right = resolve_view_paths(view, project_root)
    left_img = read_rgb(left)
    right_img = read_rgb(right)
    K = load_K_from_prediction(depth_pred_dir, view)

    c_mv = None
    if scene_views and depth_pred_root and len(scene_views) > 1:
        try:
            T_wcs = []
            T_cws = []
            depths = []
            Ks = []
            idx_map = {}
            for i, sv in enumerate(scene_views):
                pd = depth_pred_root / sv.scene_id / sv.view_id
                if not (pd / "depth_m.npy").exists():
                    continue
                try:
                    twc, tcw = world_cam_from_view(sv, object_centric=(view.dataset == "bop_tless"))
                except Exception:  # noqa: BLE001
                    continue
                T_wcs.append(twc)
                T_cws.append(tcw)
                depths.append(np.load(pd / "depth_m.npy"))
                Ks.append(load_K_from_prediction(pd, sv))
                idx_map[sv.view_id] = len(T_wcs) - 1

            if view.view_id in idx_map:
                vi = idx_map[view.view_id]
                neighbors = select_neighbor_views(vi, T_wcs, cfg.k_neighbor_views)
                if neighbors:
                    c_mv = multiview_agreement_confidence(
                        depth_m,
                        K,
                        T_wcs[vi],
                        T_cws[vi],
                        [depths[j] for j in neighbors],
                        [Ks[j] for j in neighbors],
                        [T_cws[j] for j in neighbors],
                        tau_mv_m=cfg.thresholds.tau_mv_m,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Multi-view agreement skipped for %s/%s: %s", view.scene_id, view.view_id, exc)
    else:
        logger.debug("Multi-view agreement set to 1.0 (single view or no poses) for %s/%s", view.scene_id, view.view_id)

    maps = compute_confidence_maps(
        depth_m,
        disparity,
        left_img,
        right_img,
        K,
        cfg,
        disparity_r2l=disparity_r2l,
        c_mv=c_mv,
    )
    save_confidence_maps(out_dir, maps)
    write_json(
        out_dir / "uncertainty_meta.json",
        {
            "scene_id": view.scene_id,
            "view_id": view.view_id,
            "mean_confidence": float(maps.confidence_total.mean()),
            "valid_ratio": float(maps.valid.mean()),
        },
    )

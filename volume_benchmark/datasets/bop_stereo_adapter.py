"""BOP-format rendered stereo preparation (YCB-V, T-LESS, etc.)."""

from __future__ import annotations

from pathlib import Path

from volume_benchmark.stereo.render_stereo_from_mesh import render_rectified_stereo_from_mesh
from volume_benchmark.stereo.stereo_dataset_adapter import save_prepared_stereo_scan


def prepare_bop_stereo_rendered(
    dataset_root: str | Path,
    split: str,
    object_id: int,
    out_dir: str | Path,
    baseline_m: float = 0.12,
    num_views: int = 5,
    min_visib_fract: float = 0.5,
    dataset_name: str = "bop",
) -> Path:
    """
    Select BOP views and render rectified stereo pairs from GT mesh.

    Uses tless_volume_benchmark BOP I/O when available (T-LESS); otherwise requires
    manual mesh_path and a prepared scan with poses.
    """
    root = Path(dataset_root).resolve()
    out = Path(out_dir).resolve()

    # Prefer T-LESS pipeline (same BOP layout)
    try:
        from tless_volume_benchmark.io_bop import iter_tless_candidates
        from tless_volume_benchmark.mesh_volume import (
            compute_fallback_convex_hull_gt,
            compute_mesh_volume_m3,
            discover_tless_models_dir,
            load_tless_model_mesh_meters,
        )
        from tless_volume_benchmark.view_selection import select_views

        models_path = discover_tless_models_dir(root)
        gt_mesh_tmp = out / "_gt_mesh_source.ply"
        out.mkdir(parents=True, exist_ok=True)
        mesh = load_tless_model_mesh_meters(root, object_id, model_dir=models_path.name, out_gt_mesh_path=gt_mesh_tmp)
        vol_info = compute_mesh_volume_m3(mesh, repair=False)
        if vol_info["volume_m3"] is None:
            gt_vol = compute_fallback_convex_hull_gt(mesh)
            gt_volume = {**gt_vol, "object_id": object_id, "source_mesh": str(gt_mesh_tmp), "split": split}
        else:
            gt_volume = {
                "object_id": object_id,
                "volume_m3": vol_info["volume_m3"],
                "volume_cm3": vol_info["volume_cm3"],
                "gt_type": vol_info["gt_type"],
                "watertight": vol_info["watertight"],
                "exact_gt": vol_info["gt_type"] == "mesh_watertight",
                "source_mesh": str(gt_mesh_tmp),
                "split": split,
            }

        candidates = list(
            iter_tless_candidates(root, split=split, object_id=object_id, min_visib_fract=min_visib_fract)
        )
        if not candidates:
            raise ValueError(f"No BOP candidates for object {object_id} in split {split}")
        selected = select_views(candidates, num_views=num_views)[:num_views]

        frames_data = []
        K_ref = selected[0].K
        for cand in selected:
            h, w = cand.depth_m.shape
            left, right, mask, rmeta = render_rectified_stereo_from_mesh(
                gt_mesh_tmp,
                cand.K,
                (w, h),
                cand.T_cam_to_object,
                baseline_m,
                mesh_units="m",
            )
            fmeta = {
                **rmeta,
                "source_mode": "rendered_stereo_from_bop_gt_mesh",
                "dataset": dataset_name,
                "split": split,
                "object_id": object_id,
                "scene_id": cand.scene_id,
                "image_id": cand.image_id,
            }
            use_mask = mask if mask is not None else cand.mask
            frames_data.append((left, right, use_mask, cand.T_cam_to_object, fmeta))

        metadata = {
            "source_mode": "rendered_stereo_from_bop_gt_mesh",
            "dataset": dataset_name,
            "split": split,
            "object_id": object_id,
            "baseline_m": baseline_m,
            "num_views": len(frames_data),
            "note": "Synthetic stereo from GT mesh; NOT real dataset stereo depth.",
        }
        return save_prepared_stereo_scan(
            out, K_ref, baseline_m, frames_data, gt_mesh_tmp, gt_volume, metadata=metadata
        )
    except ImportError as exc:
        raise ImportError("T-LESS/BOP stereo requires tless_volume_benchmark installed") from exc

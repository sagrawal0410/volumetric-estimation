"""Prepare normalized T-LESS object-centric scans from BOP format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tless_volume_benchmark.io_bop import CandidateFrame, iter_tless_candidates
from tless_volume_benchmark.mesh_volume import (
    compute_fallback_convex_hull_gt,
    compute_mesh_volume_m3,
    load_tless_model_mesh_meters,
    discover_tless_models_dir,
)
from tless_volume_benchmark.view_selection import save_selected_views_json, select_views
from tless_volume_benchmark.visualize import (
    depth_to_vis,
    save_fused_points_colored,
    save_mask_overlay,
)


def _save_frame(out_frames: Path, idx: int, cand: CandidateFrame) -> None:
    prefix = f"frame_{idx:03d}"
    out_frames.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_frames / f"{prefix}_rgb.png"), cv2.cvtColor(cand.rgb, cv2.COLOR_RGB2BGR))
    np.save(out_frames / f"{prefix}_depth.npy", cand.depth_m.astype(np.float32))
    cv2.imwrite(str(out_frames / f"{prefix}_mask.png"), (cand.mask.astype(np.uint8) * 255))
    np.save(out_frames / f"{prefix}_K.npy", cand.K.astype(np.float64))
    np.save(out_frames / f"{prefix}_T_cam_to_object.npy", cand.T_cam_to_object.astype(np.float64))
    meta = {
        "object_id": cand.object_id,
        "split": cand.split,
        "scene_id": cand.scene_id,
        "image_id": cand.image_id,
        "gt_id": cand.gt_id,
        "visib_fract": cand.visib_fract,
        "valid_object_depth_pixels": cand.valid_object_depth_pixels,
        "depth_scale": cand.depth_scale,
        "rgb_path": cand.rgb_path,
        "depth_path": cand.depth_path,
        "mask_path": cand.mask_path,
    }
    with (out_frames / f"{prefix}_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def prepare_tless_scan(
    dataset_root: str | Path,
    split: str,
    object_id: int,
    out_dir: str | Path,
    num_views: int = 5,
    min_visib_fract: float = 0.85,
    min_valid_depth_pixels: int = 100,
    repair_mesh: bool = False,
    use_convex_hull_fallback: bool = True,
    model_dir: str | None = None,
    model_preference: str = "cad",
) -> Path:
    """Prepare one normalized object-centric scan."""
    root = Path(dataset_root).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    gt_mesh_path = out / "gt_mesh.ply"
    models_path = (
        root / model_dir if model_dir else discover_tless_models_dir(root, preference=model_preference)
    )
    mesh = load_tless_model_mesh_meters(
        root,
        object_id,
        model_dir=models_path.name,
        out_gt_mesh_path=gt_mesh_path,
    )
    vol_info = compute_mesh_volume_m3(mesh, repair=repair_mesh)

    exact_gt = False
    if vol_info["volume_m3"] is not None and vol_info["watertight"]:
        exact_gt = vol_info["gt_type"] == "mesh_watertight"
        gt_payload = {
            "object_id": object_id,
            "volume_m3": vol_info["volume_m3"],
            "gt_type": vol_info["gt_type"],
            "watertight": vol_info["watertight"],
            "exact_gt": exact_gt,
            "repaired": vol_info["repaired"],
            "source_mesh": str(gt_mesh_path),
            "source_models_dir": str(models_path),
            "split": split,
            "num_vertices": vol_info["num_vertices"],
            "num_faces": vol_info["num_faces"],
            "bbox_extents_m": vol_info["bbox_extents_m"],
        }
    elif use_convex_hull_fallback:
        fallback = compute_fallback_convex_hull_gt(mesh)
        gt_payload = {
            "object_id": object_id,
            "volume_m3": fallback["volume_m3"],
            "gt_type": fallback["gt_type"],
            "watertight": fallback["watertight"],
            "exact_gt": False,
            "repaired": False,
            "source_mesh": str(gt_mesh_path),
            "source_models_dir": str(models_path),
            "split": split,
            "warning": "Non-watertight model; using convex hull fallback (overestimate).",
        }
    else:
        raise ValueError(
            "T-LESS model is not watertight and convex hull fallback is disabled."
        )

    with (out / "gt_volume.json").open("w", encoding="utf-8") as f:
        json.dump(gt_payload, f, indent=2)

    candidates = list(
        iter_tless_candidates(
            root,
            split=split,
            object_id=object_id,
            min_visib_fract=min_visib_fract,
            min_valid_depth_pixels=min_valid_depth_pixels,
        )
    )
    if not candidates:
        raise ValueError(
            f"No candidate frames for object_id={object_id} in split={split!r}. "
            "Check dataset extraction, masks, and min_visib_fract."
        )

    selected = select_views(candidates, num_views=num_views)
    if len(selected) < min(2, num_views):
        raise ValueError(
            f"Only {len(selected)} views selected; need at least 2 for volume estimation."
        )

    save_selected_views_json(out / "selected_views.json", selected)

    frames_dir = out / "frames"
    debug_dir = out / "debug"
    for idx, cand in enumerate(selected):
        _save_frame(frames_dir, idx, cand)
        save_mask_overlay(debug_dir / f"frame_{idx:03d}_mask_overlay.png", cand.rgb, cand.mask)
        cv2.imwrite(
            str(debug_dir / f"frame_{idx:03d}_depth_vis.png"),
            depth_to_vis(cand.depth_m, cand.mask),
        )

    save_fused_points_colored(debug_dir / "fused_points_by_view_colored.ply", selected)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare T-LESS object-centric scan")
    parser.add_argument("--dataset_root", required=True, help="e.g. data/bop/tless/tless")
    parser.add_argument("--split", default="train_primesense")
    parser.add_argument("--object_id", type=int, required=True)
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--min_visib_fract", type=float, default=0.85)
    parser.add_argument("--min_valid_depth_pixels", type=int, default=100)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--repair_mesh", action="store_true")
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Explicit models folder name (e.g. models_cad). Default: auto-detect.",
    )
    parser.add_argument(
        "--model_preference",
        default="cad",
        choices=["cad", "eval", "reconst"],
        help="When auto-detecting, prefer models_cad (default), models_eval, or models_reconst.",
    )
    args = parser.parse_args(argv)

    out = prepare_tless_scan(
        dataset_root=args.dataset_root,
        split=args.split,
        object_id=args.object_id,
        out_dir=args.out_dir,
        num_views=args.num_views,
        min_visib_fract=args.min_visib_fract,
        min_valid_depth_pixels=args.min_valid_depth_pixels,
        repair_mesh=args.repair_mesh,
        model_dir=args.model_dir,
        model_preference=args.model_preference,
    )
    print(f"Prepared scan: {out}")


if __name__ == "__main__":
    main()

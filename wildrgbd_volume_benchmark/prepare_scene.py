"""Prepare a normalized WildRGB-D scene for sparse-view volume benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from wildrgbd_volume_benchmark.io_wildrgbd import load_scene
from wildrgbd_volume_benchmark.pointcloud_fusion import fuse_sampled_frames_to_object
from wildrgbd_volume_benchmark.pseudo_gt import compute_pseudo_gt_volume_from_full_video
from wildrgbd_volume_benchmark.view_selection import (
    build_selected_views_json,
    save_selected_views_json,
    select_sparse_views,
)
from wildrgbd_volume_benchmark.visualize import save_frame_debug, visualize_scene_coverage


def _resolve_scene_path(wildrgbd_root: Path, category: str, scene_id: str) -> Path:
    sid = scene_id if scene_id.startswith("scenes_") else f"scenes_{scene_id}"
    path = wildrgbd_root / category / "scenes" / sid
    if not path.is_dir():
        raise FileNotFoundError(f"Scene not found: {path}")
    return path


def _save_sampled_frame(frames_dir: Path, idx: int, frame, T_world_to_object: np.ndarray) -> None:
    from wildrgbd_volume_benchmark.io_wildrgbd import load_depth_m, load_mask, load_rgb

    prefix = f"frame_{idx:03d}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rgb = frame.rgb if frame.rgb is not None else load_rgb(frame.rgb_path)
    depth_m = frame.depth_m if frame.depth_m is not None else load_depth_m(frame.depth_path)
    mask = frame.mask if frame.mask is not None else load_mask(frame.mask_path)
    assert frame.K is not None and frame.T_cam_to_world is not None
    T_cam_to_object = T_world_to_object @ frame.T_cam_to_world

    cv2.imwrite(str(frames_dir / f"{prefix}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(frames_dir / f"{prefix}_depth.npy", depth_m.astype(np.float32))
    cv2.imwrite(str(frames_dir / f"{prefix}_mask.png"), (mask.astype(np.uint8) * 255))
    np.save(frames_dir / f"{prefix}_K.npy", frame.K.astype(np.float64))
    np.save(frames_dir / f"{prefix}_T_cam_to_world.npy", frame.T_cam_to_world.astype(np.float64))
    np.save(frames_dir / f"{prefix}_T_cam_to_object.npy", T_cam_to_object.astype(np.float64))
    meta = {
        "frame_id": frame.frame_id,
        "rgb_path": frame.rgb_path,
        "depth_path": frame.depth_path,
        "mask_path": frame.mask_path,
    }
    with (frames_dir / f"{prefix}_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def prepare_scene(
    wildrgbd_root: str | Path,
    category: str,
    scene_id: str,
    out_dir: str | Path,
    num_views: int = 5,
    scene_types: tuple[str, ...] = ("single",),
    gt_frame_stride: int = 1,
    max_frames_for_gt: int | None = None,
    sample_min_angle_deg: float = 20.0,
    require_valid_depth_pixels: int = 1000,
    repair_mesh: bool = False,
) -> Path:
    root = Path(wildrgbd_root).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    scene_path = _resolve_scene_path(root, category, scene_id)
    from wildrgbd_volume_benchmark.io_wildrgbd import load_types_json

    types_map = load_types_json(root / category)
    key = scene_path.name.replace("scenes_", "")
    scene_type = types_map.get(key, types_map.get(scene_path.name, "unknown"))
    if scene_types and scene_type not in scene_types:
        raise ValueError(
            f"Scene {scene_path.name} has type {scene_type!r}, not in {scene_types}. "
            "Use --scene_types to include it."
        )

    scene = load_scene(scene_path, category=category, scene_id=scene_path.name, scene_type=scene_type)

    pseudo_dir = out / "pseudo_gt"
    pg = compute_pseudo_gt_volume_from_full_video(
        scene,
        pseudo_dir,
        frame_stride=gt_frame_stride,
        max_frames_for_gt=max_frames_for_gt,
        repair_mesh=repair_mesh,
    )
    T_world_to_object = np.array(pg["T_world_to_object"], dtype=np.float64)

    selected = select_sparse_views(
        scene,
        T_world_to_object,
        num_views=num_views,
        min_angle_deg=sample_min_angle_deg,
        require_valid_depth_pixels=require_valid_depth_pixels,
    )

    sampled_dir = out / "sampled_5view"
    sv_payload = build_selected_views_json(scene, selected, T_world_to_object)
    save_selected_views_json(sampled_dir / "selected_views.json", sv_payload)

    frames_dir = sampled_dir / "frames"
    debug_dir = out / "debug"
    camera_centers = []
    for idx, frame in enumerate(selected):
        _save_sampled_frame(frames_dir, idx, frame, T_world_to_object)
        from wildrgbd_volume_benchmark.io_wildrgbd import load_depth_m, load_mask, load_rgb

        if frame.rgb is None:
            frame.rgb = load_rgb(frame.rgb_path)
            frame.depth_m = load_depth_m(frame.depth_path)
            frame.mask = load_mask(frame.mask_path)
        save_frame_debug(frame, debug_dir, f"frame_{idx:03d}")
        T_cam_to_object = T_world_to_object @ frame.T_cam_to_world
        camera_centers.append(T_cam_to_object[:3, 3])

    sampled_pts = fuse_sampled_frames_to_object(scene, [f.frame_id for f in selected], T_world_to_object)
    import trimesh

    trimesh.PointCloud(sampled_pts).export(debug_dir / "sampled_fused_pointcloud.ply")

    full_ply = pseudo_dir / "full_fused_pointcloud.ply"
    if full_ply.is_file():
        full_pts = np.asarray(trimesh.load(full_ply).vertices)
        visualize_scene_coverage(full_pts, camera_centers, debug_dir / "scene_coverage.png")

    meta = {
        "category": category,
        "scene_id": scene_path.name,
        "scene_type": scene_type,
        "num_sampled_views": len(selected),
        "T_world_to_object": T_world_to_object.tolist(),
        "pseudo_gt_method": pg.get("chosen_method"),
        "pseudo_gt_volume_cm3": pg.get("volume_cm3"),
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare WildRGB-D scene for volume benchmark")
    parser.add_argument("--wildrgbd_root", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--scene_id", required=True, help="e.g. scenes_000123")
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--scene_types", default="single")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--gt_frame_stride", type=int, default=1)
    parser.add_argument("--max_frames_for_gt", type=int, default=None)
    parser.add_argument("--sample_min_angle_deg", type=float, default=20.0)
    parser.add_argument("--require_valid_depth_pixels", type=int, default=1000)
    parser.add_argument("--repair_mesh", action="store_true")
    args = parser.parse_args(argv)

    types = tuple(t.strip() for t in args.scene_types.split(",") if t.strip())
    out = prepare_scene(
        wildrgbd_root=args.wildrgbd_root,
        category=args.category,
        scene_id=args.scene_id,
        out_dir=args.out_dir,
        num_views=args.num_views,
        scene_types=types,
        gt_frame_stride=args.gt_frame_stride,
        max_frames_for_gt=args.max_frames_for_gt,
        sample_min_angle_deg=args.sample_min_angle_deg,
        require_valid_depth_pixels=args.require_valid_depth_pixels,
        repair_mesh=args.repair_mesh,
    )
    print(f"Prepared scene: {out}")


if __name__ == "__main__":
    main()

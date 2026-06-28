"""WildRGB-D rendered stereo preparation from pseudo-GT mesh."""

from __future__ import annotations

from pathlib import Path

from volume_benchmark.stereo.render_stereo_from_mesh import render_rectified_stereo_from_mesh
from volume_benchmark.stereo.stereo_dataset_adapter import save_prepared_stereo_scan


def prepare_wildrgbd_stereo_rendered(
    prepared_scene_dir: str | Path,
    out_dir: str | Path,
    baseline_m: float = 0.12,
) -> Path:
    """
    Build stereo scan from an existing WildRGB-D prepared scene (sampled views + pseudo-GT mesh).

    WildRGB-D provides monocular RGB-D only — real stereo pairs are NOT available.
    This renders rectified pairs from pseudo_gt/gt mesh.
    """
    from wildrgbd_volume_benchmark.scan_io import load_prepared_scene

    scene = load_prepared_scene(prepared_scene_dir)
    gt_mesh = scene.scene_dir / "pseudo_gt" / "full_tsdf_mesh.ply"
    if not gt_mesh.is_file():
        gt_mesh = scene.scene_dir / "pseudo_gt" / "full_fused_pointcloud.ply"
    if not gt_mesh.is_file():
        raise FileNotFoundError(
            f"No renderable mesh under {scene.scene_dir}/pseudo_gt/. "
            "Run wildrgbd prepare_scene first."
        )

    # Point clouds cannot render stereo — need mesh
    if gt_mesh.suffix == ".ply" and "pointcloud" in gt_mesh.name.lower():
        raise ValueError(
            "WildRGB-D only has fused point cloud, not a mesh. "
            "Re-run prepare_scene with repair_mesh or ensure full_tsdf_mesh.ply exists. "
            "Cannot use FoundationStereo rendered mode without a mesh."
        )

    frames_data = []
    K_ref = scene.frames[0].K
    for sf in scene.frames:
        h, w = sf.depth_m.shape
        left, right, mask, rmeta = render_rectified_stereo_from_mesh(
            gt_mesh,
            sf.K,
            (w, h),
            sf.T_cam_to_object,
            baseline_m,
            mesh_units="m",
        )
        fmeta = {
            **rmeta,
            "source_mode": "rendered_stereo_from_wildrgbd_pseudo_gt_mesh",
            "category": scene.category,
            "scene_id": scene.scene_id,
            "frame_index": sf.index,
        }
        use_mask = mask if mask is not None else sf.mask
        frames_data.append((left, right, use_mask, sf.T_cam_to_object, fmeta))

    gt_volume = dict(scene.pseudo_gt)
    gt_volume["note"] = "pseudo_gt compared against FoundationStereo depth path"

    metadata = {
        "source_mode": "rendered_stereo_from_wildrgbd_pseudo_gt_mesh",
        "dataset": "wildrgbd",
        "category": scene.category,
        "scene_id": scene.scene_id,
        "baseline_m": baseline_m,
        "prepared_scene_dir": str(scene.scene_dir),
        "note": "Monocular WildRGB-D — stereo is synthetic from pseudo-GT mesh.",
    }
    return save_prepared_stereo_scan(
        out_dir,
        K_ref,
        baseline_m,
        frames_data,
        gt_mesh,
        gt_volume,
        metadata=metadata,
    )

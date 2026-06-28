"""Pseudo ground-truth volume from full WildRGB-D video reconstruction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import trimesh

from wildrgbd_volume_benchmark.geometry import (
    estimate_object_frame_from_full_points,
    invert_T,
    make_o3d_intrinsic,
    transform_points,
)
from wildrgbd_volume_benchmark.io_wildrgbd import WildRGBDScene, iter_scene_frames
from wildrgbd_volume_benchmark.pointcloud_fusion import fuse_scene_pointcloud_full


def _clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    m.remove_infinite_values()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    m.fix_normals()
    return m


def _voxel_occupancy_volume(points: np.ndarray, voxel_size: float) -> float:
    if points.shape[0] == 0:
        return 0.0
    mins = points.min(axis=0)
    coords = np.floor((points - mins) / voxel_size).astype(np.int64)
    unique = np.unique(coords, axis=0)
    return float(unique.shape[0] * (voxel_size ** 3))


def _alpha_shape_mesh(points: np.ndarray, alpha: float):
    if not _open3d_available():
        return None
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        if len(mesh_o3d.triangles) == 0:
            return None
        return _clean_mesh(
            trimesh.Trimesh(
                vertices=np.asarray(mesh_o3d.vertices),
                faces=np.asarray(mesh_o3d.triangles),
                process=False,
            )
        )
    except Exception:
        return None


def _open3d_available() -> bool:
    if os.environ.get("WILDRGBD_SKIP_OPEN3D", "0") == "1":
        return False
    try:
        import open3d as o3d  # noqa: F401
        return True
    except Exception:
        return False


def _tsdf_mesh_volume_object(
    scene: WildRGBDScene,
    T_world_to_object: np.ndarray,
    frame_stride: int,
    max_frames: int | None,
    voxel_length: float,
    sdf_trunc: float,
    depth_trunc: float = 5.0,
) -> tuple[trimesh.Trimesh | None, bool, float | None]:
    if not _open3d_available():
        return None, False, None
    import open3d as o3d

    if scene.image_size is None or scene.K is None:
        raise ValueError("Scene must have K and image_size for TSDF")

    width, height = scene.image_size
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    count = 0
    for frame in iter_scene_frames(scene, frame_stride=frame_stride, max_frames=max_frames):
        assert frame.K is not None and frame.T_cam_to_world is not None
        depth = frame.depth_m.copy()
        depth[~frame.mask] = 0.0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8)),
            o3d.geometry.Image(depth.astype(np.float32)),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        T_cam_to_object = T_world_to_object @ frame.T_cam_to_world
        T_object_to_cam = invert_T(T_cam_to_object)
        intrinsic = make_o3d_intrinsic(frame.K, width, height)
        volume.integrate(rgbd, intrinsic, T_object_to_cam)
        count += 1

    if count == 0:
        return None, False, None

    mesh_o3d = volume.extract_triangle_mesh()
    if len(mesh_o3d.triangles) == 0:
        return None, False, None
    mesh = _clean_mesh(
        trimesh.Trimesh(
            vertices=np.asarray(mesh_o3d.vertices),
            faces=np.asarray(mesh_o3d.triangles),
            process=False,
        )
    )
    watertight = bool(mesh.is_watertight)
    vol = abs(float(mesh.volume)) if watertight and mesh.volume > 0 else None
    return mesh, watertight, vol


def compute_pseudo_gt_volume_from_full_video(
    scene: WildRGBDScene,
    output_dir: str | Path,
    frame_stride: int = 1,
    max_frames_for_gt: int | None = None,
    voxel_downsample: float = 0.002,
    tsdf_voxel_length: float = 0.0025,
    tsdf_sdf_trunc: float = 0.0125,
    alpha_shape_alphas: Sequence[float] = (0.01, 0.02, 0.04, 0.08),
    voxel_occupancy_size: float = 0.0025,
    repair_mesh: bool = False,
) -> dict[str, Any]:
    """
    Compute pseudo-GT volume from full video reconstruction.

    Never treat result as exact scalar GT.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    points_world = fuse_scene_pointcloud_full(
        scene,
        frame_stride=frame_stride,
        max_frames=max_frames_for_gt,
        voxel_downsample=voxel_downsample,
    )
    trimesh.PointCloud(points_world).export(out / "full_fused_pointcloud.ply")

    T_world_to_object = estimate_object_frame_from_full_points(points_world)
    points_obj = transform_points(points_world, T_world_to_object)

    chosen_method = "full_voxel_occupancy_volume"
    volume_m3: float | None = None
    watertight = False
    mesh_out: trimesh.Trimesh | None = None

    # Method 1: full TSDF
    tsdf_mesh, tsdf_wt, tsdf_vol = _tsdf_mesh_volume_object(
        scene,
        T_world_to_object,
        frame_stride,
        max_frames_for_gt,
        tsdf_voxel_length,
        tsdf_sdf_trunc,
    )
    if tsdf_mesh is not None:
        tsdf_mesh.export(out / "full_tsdf_mesh.ply")
        mesh_out = tsdf_mesh

    if tsdf_vol is not None and tsdf_wt:
        chosen_method = "full_tsdf_mesh_volume"
        volume_m3 = tsdf_vol
        watertight = True
    elif tsdf_mesh is not None and repair_mesh:
        try:
            trimesh.repair.fill_holes(tsdf_mesh)
            tsdf_mesh.fix_normals()
            tsdf_mesh.merge_vertices()
            if tsdf_mesh.is_watertight and tsdf_mesh.volume > 0:
                chosen_method = "full_tsdf_mesh_volume_repaired"
                volume_m3 = abs(float(tsdf_mesh.volume))
                watertight = True
                mesh_out = tsdf_mesh
                tsdf_mesh.export(out / "full_tsdf_mesh.ply")
        except Exception as exc:
            warnings.append(f"TSDF repair failed: {exc}")

    # Method 2: alpha shape on fused cloud
    if volume_m3 is None:
        for alpha in alpha_shape_alphas:
            amesh = _alpha_shape_mesh(points_obj, alpha)
            if amesh is not None and amesh.is_watertight and amesh.volume > 0:
                chosen_method = "full_pointcloud_alpha_shape_volume"
                volume_m3 = abs(float(amesh.volume))
                watertight = True
                mesh_out = amesh
                amesh.export(out / "full_alpha_shape_mesh.ply")
                break
        if volume_m3 is None:
            warnings.append("Alpha shape did not yield watertight mesh")

    # Method 3: voxel occupancy fallback
    if volume_m3 is None:
        chosen_method = "full_voxel_occupancy_volume"
        volume_m3 = _voxel_occupancy_volume(points_obj, voxel_occupancy_size)
        watertight = False
        warnings.append("Using coarse voxel occupancy pseudo-GT (not watertight mesh volume)")

    num_frames = len(list(iter_scene_frames(scene, frame_stride=frame_stride, max_frames=max_frames_for_gt)))

    payload: dict[str, Any] = {
        "gt_type": "full_video_reconstruction_pseudo_gt",
        "chosen_method": chosen_method,
        "volume_m3": volume_m3,
        "volume_cm3": volume_m3 * 1e6 if volume_m3 is not None else None,
        "watertight": watertight,
        "exact_gt": False,
        "num_full_frames_used": num_frames,
        "frame_stride": frame_stride,
        "category": scene.category,
        "scene_id": scene.scene_id,
        "scene_type": scene.scene_type,
        "warnings": warnings,
        "T_world_to_object": T_world_to_object.tolist(),
    }

    with (out / "pseudo_gt_volume.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload

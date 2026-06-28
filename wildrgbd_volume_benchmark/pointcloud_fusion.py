"""Full-scene and sampled point cloud fusion for WildRGB-D."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import trimesh

from wildrgbd_volume_benchmark.geometry import backproject_depth, transform_points
from wildrgbd_volume_benchmark.io_wildrgbd import WildRGBDScene, iter_scene_frames


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return points[np.sort(idx)]


def _remove_statistical_outliers(points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    if points.shape[0] <= nb_neighbors:
        return points
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        dists, _ = tree.query(points, k=min(nb_neighbors + 1, points.shape[0]))
        mean_dist = dists[:, 1:].mean(axis=1)
        thresh = float(mean_dist.mean() + std_ratio * mean_dist.std())
        return points[mean_dist <= thresh]
    except Exception:
        return points


def _largest_cluster(points: np.ndarray, eps: float, min_samples: int = 10) -> np.ndarray:
    if points.shape[0] < min_samples:
        return points
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return points
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    valid = labels[labels >= 0]
    if valid.size == 0:
        return points
    unique, counts = np.unique(valid, return_counts=True)
    best = unique[int(np.argmax(counts))]
    return points[labels == best]


def fuse_scene_pointcloud_full(
    scene: WildRGBDScene,
    frame_stride: int = 1,
    max_frames: int | None = None,
    voxel_downsample: float = 0.0025,
    remove_outliers: bool = True,
) -> np.ndarray:
    """Fuse masked depth from many frames into a world-frame point cloud."""
    chunks: list[np.ndarray] = []
    for frame in iter_scene_frames(scene, frame_stride=frame_stride, max_frames=max_frames):
        assert frame.K is not None and frame.T_cam_to_world is not None
        assert frame.depth_m is not None and frame.mask is not None
        pts = backproject_depth(frame.depth_m, frame.mask, frame.K, frame.T_cam_to_world)
        if pts.size:
            chunks.append(pts)
    if not chunks:
        raise ValueError(f"No points fused for scene {scene.scene_id}")
    points = np.vstack(chunks)
    points = _voxel_downsample(points, voxel_downsample)
    if remove_outliers:
        points = _remove_statistical_outliers(points)
        points = _largest_cluster(points, eps=3.0 * voxel_downsample)
    return points


def fuse_sampled_frames_to_object(
    scene: WildRGBDScene,
    selected_frame_ids: Sequence[str],
    T_world_to_object: np.ndarray,
    voxel_downsample: float = 0.0025,
) -> np.ndarray:
    id_set = set(selected_frame_ids)
    chunks: list[np.ndarray] = []
    for frame in scene.frames:
        if frame.frame_id not in id_set:
            continue
        if frame.depth_m is None:
            from wildrgbd_volume_benchmark.io_wildrgbd import load_depth_m, load_mask

            frame.depth_m = load_depth_m(frame.depth_path)
            frame.mask = load_mask(frame.mask_path)
        assert frame.K is not None and frame.T_cam_to_world is not None
        assert frame.depth_m is not None and frame.mask is not None
        pts_w = backproject_depth(frame.depth_m, frame.mask, frame.K, frame.T_cam_to_world)
        if pts_w.size:
            chunks.append(transform_points(pts_w, T_world_to_object))
    if not chunks:
        raise ValueError("No points from sampled frames")
    points = np.vstack(chunks)
    return _voxel_downsample(points, voxel_downsample)


def save_colored_by_frame_pointcloud(
    path: str,
    scene: WildRGBDScene,
    frame_ids: Sequence[str],
    colors: list[list[int]] | None = None,
) -> None:
    from wildrgbd_volume_benchmark.io_wildrgbd import load_depth_m, load_mask

    colors = colors or [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]]
    all_pts = []
    all_cols = []
    for i, fid in enumerate(frame_ids):
        frame = next(f for f in scene.frames if f.frame_id == fid)
        if frame.depth_m is None:
            frame.depth_m = load_depth_m(frame.depth_path)
            frame.mask = load_mask(frame.mask_path)
        assert frame.K is not None and frame.T_cam_to_world is not None
        pts = backproject_depth(frame.depth_m, frame.mask, frame.K, frame.T_cam_to_world)
        if pts.size:
            all_pts.append(pts)
            col = colors[i % len(colors)]
            all_cols.append(np.tile(col, (pts.shape[0], 1)))
    cloud = trimesh.PointCloud(np.vstack(all_pts), colors=np.vstack(all_cols))
    cloud.export(path)

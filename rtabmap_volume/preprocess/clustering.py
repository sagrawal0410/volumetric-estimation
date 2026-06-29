"""DBSCAN clustering of point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.cluster import DBSCAN

from rtabmap_volume.config import ClusteringConfig


@dataclass
class ClusterResult:
    labels: np.ndarray
    selected_cluster_id: int
    summary_df: pd.DataFrame
    colored_pcd: o3d.geometry.PointCloud


def _cluster_color(label: int) -> list[float]:
    if label < 0:
        return [0.3, 0.3, 0.3]
    rng = np.random.default_rng(label + 42)
    return rng.random(3).tolist()


def cluster_point_cloud(
    pcd: o3d.geometry.PointCloud,
    cfg: ClusteringConfig,
    roi_center: np.ndarray | None = None,
    cluster_id: int | None = None,
    seed: int = 42,
) -> ClusterResult:
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return ClusterResult(
            labels=np.array([]),
            selected_cluster_id=-1,
            summary_df=pd.DataFrame(),
            colored_pcd=o3d.geometry.PointCloud(),
        )

    db = DBSCAN(eps=cfg.eps_m, min_samples=cfg.min_points)
    labels = db.fit_predict(pts)

    rows = []
    unique = sorted(set(labels) - {-1})
    for cid in unique:
        mask = labels == cid
        cluster_pts = pts[mask]
        mn = cluster_pts.min(axis=0)
        mx = cluster_pts.max(axis=0)
        dims = mx - mn
        rows.append(
            {
                "cluster_id": cid,
                "num_points": int(mask.sum()),
                "bbox_dims_x": dims[0],
                "bbox_dims_y": dims[1],
                "bbox_dims_z": dims[2],
                "bbox_volume": float(np.prod(dims)),
                "centroid_x": cluster_pts.mean(axis=0)[0],
                "centroid_y": cluster_pts.mean(axis=0)[1],
                "centroid_z": cluster_pts.mean(axis=0)[2],
            }
        )
    summary_df = pd.DataFrame(rows)

    if cluster_id is not None:
        selected = cluster_id
    elif cfg.choose_cluster == "nearest_to_roi_center" and roi_center is not None and len(rows):
        centroids = summary_df[["centroid_x", "centroid_y", "centroid_z"]].values
        dists = np.linalg.norm(centroids - roi_center, axis=1)
        selected = int(summary_df.iloc[int(np.argmin(dists))]["cluster_id"])
    elif len(rows):
        selected = int(summary_df.loc[summary_df["num_points"].idxmax(), "cluster_id"])
    else:
        selected = -1

    colored = o3d.geometry.PointCloud(pcd)
    colors = np.array([_cluster_color(int(l)) for l in labels])
    colored.colors = o3d.utility.Vector3dVector(colors)

    return ClusterResult(labels=labels, selected_cluster_id=selected, summary_df=summary_df, colored_pcd=colored)


def extract_cluster(pcd: o3d.geometry.PointCloud, labels: np.ndarray, cluster_id: int) -> o3d.geometry.PointCloud:
    if cluster_id < 0:
        return pcd
    mask = labels == cluster_id
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask])
    if pcd.has_colors():
        cropped.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        cropped.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])
    return cropped


def save_cluster_summary(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

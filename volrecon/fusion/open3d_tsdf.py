"""Open3D ScalableTSDFVolume plain baseline reconstructor."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh
import yaml

from volrecon.fusion.fusion_utils import extrinsic_for_open3d, open3d_intrinsic
from volrecon.io.image_io import read_rgb
from volrecon.io.json_io import write_json

logger = logging.getLogger(__name__)


@dataclass
class PlainTSDFConfig:
    voxel_length_m: float = 0.003
    sdf_trunc_m: float = 0.015
    depth_trunc_m: float = 2.0
    color_type: str = "RGB8"
    min_depth_m: float = 0.1
    max_depth_m: float = 2.0
    max_views: int | None = None
    frame_stride: int = 1
    integrate_color: bool = True
    mesh_cleanup: bool = True
    keep_largest_component: bool = True


class PlainTSDFReconstructor:
    def __init__(self, cfg: PlainTSDFConfig, bounds: np.ndarray) -> None:
        self.cfg = cfg
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self._volume = self._make_volume()
        self._frames: list[dict[str, Any]] = []

    def _make_volume(self) -> o3d.pipelines.integration.ScalableTSDFVolume:
        color_type = (
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            if self.cfg.color_type == "RGB8" and self.cfg.integrate_color
            else o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        )
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(self.cfg.voxel_length_m),
            sdf_trunc=float(self.cfg.sdf_trunc_m),
            color_type=color_type,
        )

    def integrate_view(
        self,
        rgb_path: Path | None,
        depth_m_path: Path,
        K: np.ndarray,
        T_world_cam: np.ndarray | None = None,
        T_cam_world: np.ndarray | None = None,
        view_id: str | None = None,
    ) -> None:
        depth_m = np.load(depth_m_path).astype(np.float64)
        depth_m = depth_m.copy()
        depth_m[depth_m < self.cfg.min_depth_m] = 0.0
        depth_m[depth_m > self.cfg.max_depth_m] = 0.0
        depth_m[depth_m > self.cfg.depth_trunc_m] = 0.0

        h, w = depth_m.shape
        intrinsic = open3d_intrinsic(K, w, h)
        extrinsic = extrinsic_for_open3d(T_world_cam, T_cam_world)

        depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
        if rgb_path is not None and self.cfg.integrate_color:
            rgb = read_rgb(rgb_path)
            if rgb.shape[:2] != (h, w):
                import cv2

                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=float(self.cfg.depth_trunc_m),
                convert_rgb_to_intensity=False,
            )
        else:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8)),
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=float(self.cfg.depth_trunc_m),
                convert_rgb_to_intensity=False,
            )

        self._volume.integrate(rgbd, intrinsic, extrinsic)
        self._frames.append(
            {
                "view_id": view_id,
                "depth_m_path": str(depth_m_path),
                "rgb_path": str(rgb_path) if rgb_path else None,
            }
        )

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        return self._volume.extract_triangle_mesh()

    def extract_pointcloud(self) -> o3d.geometry.PointCloud:
        return self._volume.extract_point_cloud()

    def save_outputs(self, out_dir: Path) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh_raw = self.extract_mesh()
        mesh_raw.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out_dir / "mesh_raw.ply"), mesh_raw)

        mesh_clean = self._cleanup_mesh(mesh_raw)
        o3d.io.write_triangle_mesh(str(out_dir / "mesh_clean.ply"), mesh_clean)

        pcd = self.extract_pointcloud()
        o3d.io.write_point_cloud(str(out_dir / "fused_pointcloud.ply"), pcd)

        with (out_dir / "tsdf_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self.cfg), f)

        write_json(
            out_dir / "frame_list.json",
            {"frames": self._frames, "bounds": {"min": self.bounds[0].tolist(), "max": self.bounds[1].tolist()}},
        )
        save_bounds_json = out_dir / "bounds.json"
        write_json(
            save_bounds_json,
            {"min_xyz": self.bounds[0].tolist(), "max_xyz": self.bounds[1].tolist(), "source": "fusion"},
        )

        return {
            "mesh_raw": out_dir / "mesh_raw.ply",
            "mesh_clean": out_dir / "mesh_clean.ply",
            "fused_pointcloud": out_dir / "fused_pointcloud.ply",
        }

    def _cleanup_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        if not self.cfg.mesh_cleanup:
            return mesh
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        try:
            mesh.remove_non_manifold_edges()
        except Exception:  # noqa: BLE001
            pass
        mesh.compute_vertex_normals()

        if self.cfg.keep_largest_component:
            triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
            if len(cluster_n_triangles) > 0:
                largest = int(np.argmax(cluster_n_triangles))
                mask = np.asarray(triangle_clusters) == largest
                mesh = mesh.select_by_index(np.where(mask)[0], cleanup=True)
                mesh.compute_vertex_normals()
        return mesh


def o3d_mesh_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False,
    )

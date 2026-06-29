"""Main volume estimation pipeline."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh
from rich.console import Console
from rich.progress import Progress

from rtabmap_volume.config import PipelineConfig, load_config, save_config
from rtabmap_volume.io.load_geometry import (
    GeometryType,
    LoadedGeometry,
    load_geometry,
    mesh_to_dense_point_cloud,
    inspect_geometry_stats,
)
from rtabmap_volume.io.save_outputs import (
    append_log_step,
    copy_input,
    ensure_run_dirs,
    init_processing_log,
    save_json,
    save_mesh,
    save_point_cloud,
    save_voxel_grid,
    save_volume_csv,
    save_warnings,
    save_yaml,
)
from rtabmap_volume.mesh.mesh_cleaning import clean_mesh
from rtabmap_volume.mesh.mesh_from_pointcloud import reconstruct_alpha_shape, reconstruct_ball_pivoting, reconstruct_poisson
from rtabmap_volume.mesh.mesh_repair import repair_mesh
from rtabmap_volume.preprocess.denoise import denoise_point_cloud
from rtabmap_volume.preprocess.scale_units import apply_scaling
from rtabmap_volume.preprocess.segment_object import segment_geometry
from rtabmap_volume.volume.alpha_shape_volume import compute_alpha_shape_volume
from rtabmap_volume.volume.convex_hull_volume import compute_convex_hull_volume
from rtabmap_volume.volume.consensus import compute_consensus
from rtabmap_volume.volume.heightfield_volume import compute_heightfield_volume
from rtabmap_volume.volume.mesh_volume import VolumeEstimate, compute_mesh_volume
from rtabmap_volume.volume.voxel_volume import compute_voxel_volumes
from rtabmap_volume.viz.html_report import generate_html_report
from rtabmap_volume.viz.plots import plot_alpha_volumes, plot_volume_methods
from rtabmap_volume.viz.screenshots import render_geometry_screenshot

console = Console()


class VolumePipeline:
    def __init__(
        self,
        input_path: Path,
        out_dir: Path,
        config: PipelineConfig,
        units: str = "m",
        segmentation: str | None = None,
        roi_json: str | None = None,
        known_scale_json: str | None = None,
        overwrite: bool = False,
        command: str = "",
    ) -> None:
        self.input_path = Path(input_path)
        self.out_dir = Path(out_dir)
        self.config = config
        self.units = units
        self.roi_json = roi_json
        self.known_scale_json = known_scale_json
        self.overwrite = overwrite
        self.command = command
        self.warnings: list[str] = []
        self.seed = config.output.seed

        if segmentation:
            self.config.segmentation.mode = segmentation

    def _check_overwrite(self) -> None:
        if self.out_dir.exists() and any(self.out_dir.iterdir()) and not self.overwrite:
            raise FileExistsError(
                f"Output directory {self.out_dir} exists. Pass --overwrite to replace."
            )

    def run(self) -> dict[str, Any]:
        self._check_overwrite()
        dirs = ensure_run_dirs(self.out_dir)
        log = init_processing_log(self.command)
        np.random.seed(self.seed)

        with Progress() as progress:
            task = progress.add_task("[cyan]Volume pipeline...", total=8)

            # Load
            geom = load_geometry(self.input_path)
            self.warnings.extend(geom.load_warnings or [])
            copy_input(self.input_path, dirs["inputs"] / f"copied_input_geometry{self.input_path.suffix}")
            save_yaml(self.config.to_dict(), dirs["inputs"] / "config_used.yaml")
            append_log_step(log, "load", inspect_geometry_stats(geom))
            progress.advance(task)

            mesh = geom.mesh
            pcd = geom.point_cloud

            # Scale
            scale_result = apply_scaling(mesh, pcd, self.units, self.known_scale_json)
            self.warnings.extend(scale_result.warnings)
            if mesh is None and pcd is not None:
                save_point_cloud(pcd, dirs["processed"] / "cloud_raw.ply")
            elif mesh is not None:
                save_mesh(mesh, dirs["processed"] / "cloud_raw.ply")
                if pcd is None:
                    pcd = mesh_to_dense_point_cloud(mesh)
            progress.advance(task)

            # Denoise cloud
            if pcd is not None:
                pcd_raw = copy.deepcopy(pcd)
                pcd = denoise_point_cloud(pcd, self.config.denoise)
                save_point_cloud(pcd_raw, dirs["processed"] / "cloud_raw.ply")
                save_point_cloud(pcd, dirs["processed"] / "cloud_denoised.ply")
            progress.advance(task)

            # Segment
            seg = segment_geometry(mesh, pcd, self.config, self.roi_json, seed=self.seed)
            self.warnings.extend(seg.warnings)
            mesh = seg.mesh
            pcd = seg.point_cloud
            save_point_cloud(pcd, dirs["processed"] / "cloud_cropped.ply")
            save_point_cloud(pcd, dirs["processed"] / "cloud_object_segmented.ply")
            save_json(seg.diagnostics, dirs["logs"] / "segmentation_diagnostics.json")
            append_log_step(log, "segment", seg.diagnostics)
            progress.advance(task)

            # Mesh cleaning / repair
            clean_mesh_tm: trimesh.Trimesh | None = None
            repaired_mesh: trimesh.Trimesh | None = None
            repair_report = None

            if mesh is not None and len(mesh.faces) > 0:
                clean_mesh_tm, before_q, after_q = clean_mesh(mesh, self.config.mesh_cleaning)
                save_mesh(clean_mesh_tm, dirs["processed"] / "mesh_input_clean.ply")
                save_mesh(clean_mesh_tm, dirs["processed"] / "mesh_clean_unrepaired.ply")
                save_mesh(clean_mesh_tm, dirs["processed"] / "mesh_largest_component.ply")
                repaired_mesh, repair_report = repair_mesh(clean_mesh_tm, self.config.mesh_repair)
                save_mesh(repaired_mesh, dirs["processed"] / "mesh_repaired.ply")
                mesh = clean_mesh_tm

            poisson_mesh: trimesh.Trimesh | None = None
            bpa_mesh: trimesh.Trimesh | None = None
            alpha_mesh: trimesh.Trimesh | None = None

            if self.config.reconstruct_mesh_from_cloud and pcd is not None and len(pcd.points) > 0:
                if self.config.run_poisson:
                    poisson = reconstruct_poisson(pcd, self.config.poisson)
                    self.warnings.extend(poisson.warnings)
                    poisson_mesh = poisson.mesh
                    save_mesh(poisson_mesh, dirs["processed"] / "mesh_reconstructed_poisson.ply")
                if self.config.run_bpa:
                    bpa = reconstruct_ball_pivoting(pcd, self.config.ball_pivoting)
                    self.warnings.extend(bpa.warnings)
                    bpa_mesh = bpa.mesh
                    save_mesh(bpa_mesh, dirs["processed"] / "mesh_reconstructed_bpa.ply")
                if self.config.run_alpha_shape:
                    alpha_est, alpha_mesh, alpha_vols = compute_alpha_shape_volume(pcd, self.config.alpha_shape)
                    save_mesh(alpha_mesh, dirs["processed"] / "mesh_alpha_shape.ply")
                    plot_alpha_volumes(alpha_vols, dirs["screenshots"] / "volume_vs_alpha.png")
            progress.advance(task)

            # Volume estimates
            estimates: dict[str, VolumeEstimate] = {}
            estimates["direct_mesh_volume"] = compute_mesh_volume(clean_mesh_tm or mesh, "direct_mesh_volume")
            estimates["repaired_mesh_volume"] = compute_mesh_volume(repaired_mesh, "repaired_mesh_volume")
            if repair_report and clean_mesh_tm and repair_report.changes.get("face_delta", 0) > len(clean_mesh_tm.faces) * 0.5:
                estimates["repaired_mesh_volume"].reliable = False
                estimates["repaired_mesh_volume"].warnings.append("Repair changed topology significantly")

            estimates["poisson_mesh_volume"] = compute_mesh_volume(poisson_mesh, "poisson_mesh_volume")
            estimates["ball_pivoting_mesh_volume"] = compute_mesh_volume(bpa_mesh, "ball_pivoting_mesh_volume")
            estimates["alpha_shape_volume"] = compute_mesh_volume(alpha_mesh, "alpha_shape_volume")

            voxel_est, voxel_grids = compute_voxel_volumes(pcd, clean_mesh_tm or mesh, self.config.voxel)
            estimates["voxel_occupancy_volume"] = voxel_est
            if voxel_grids:
                finest = min(voxel_grids.keys())
                vg = voxel_grids[finest]
                save_voxel_grid(vg.occupied, vg.voxel_size, vg.origin, dirs["processed"] / "voxel_grid.npz")

            estimates["convex_hull_volume"] = compute_convex_hull_volume(pcd, clean_mesh_tm or mesh)

            plane_model = seg.diagnostics.get("plane_model")
            if pcd is not None and (
                self.config.segmentation.mode == "height_above_plane"
                or self.config.consensus.pile_mode
            ):
                hf = compute_heightfield_volume(
                    pcd,
                    self.config.heightfield,
                    plane_model=np.array(plane_model) if plane_model else None,
                    plane_cfg=self.config.plane_removal,
                )
                estimates["heightfield_volume"] = hf
            progress.advance(task)

            # Consensus
            context = {
                "mesh_watertight": bool(clean_mesh_tm.is_watertight) if clean_mesh_tm else False,
                "scale_warnings": scale_result.warnings,
                "segmentation_ambiguous": len(seg.diagnostics.get("cluster_summary", [])) > 2,
            }
            consensus = compute_consensus(estimates, self.config.consensus, context)
            self.warnings.extend(consensus.warnings)
            volume_dict = consensus.to_dict()
            save_json(volume_dict, dirs["reports"] / "volume.json")
            save_volume_csv(volume_dict, dirs["reports"] / "volume.csv")
            progress.advance(task)

            # Visualizations
            if self.config.output.generate_screenshots:
                render_geometry_screenshot(mesh=clean_mesh_tm, pcd=pcd, out_path=dirs["screenshots"] / "raw_geometry.png")
                render_geometry_screenshot(pcd=pcd, out_path=dirs["screenshots"] / "cropped_geometry.png")
                render_geometry_screenshot(pcd=pcd, out_path=dirs["screenshots"] / "segmented_object.png")
                render_geometry_screenshot(mesh=repaired_mesh, out_path=dirs["screenshots"] / "repaired_mesh.png")
            plot_volume_methods(volume_dict["all_estimates"], dirs["screenshots"] / "volume_methods_barplot.png")

            if self.config.output.generate_html_report:
                generate_html_report(
                    dirs["reports"] / "report.html",
                    {
                        "input_path": str(self.input_path),
                        "command": self.command,
                        "config_path": str(dirs["inputs"] / "config_used.yaml"),
                        "geometry_stats": inspect_geometry_stats(geom),
                        "final_volume_m3": consensus.final_volume_m3,
                        "final_volume_liters": consensus.final_volume_liters,
                        "confidence": consensus.confidence,
                        "confidence_score": consensus.confidence_score_0_1,
                        "recommended_estimator": consensus.recommended_estimator,
                        "upper_bound_m3": consensus.upper_bound_m3,
                        "all_estimates": consensus.all_estimates,
                        "warnings": self.warnings,
                    },
                    dirs["screenshots"],
                )
            progress.advance(task)

            save_warnings(self.warnings, dirs["logs"] / "warnings.txt")
            log["finished_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
            save_json(log, dirs["logs"] / "processing_log.json")
            progress.advance(task)

        console.print(f"[green]Done.[/green] Final volume: {consensus.final_volume_m3} m³ ({consensus.confidence} confidence)")
        return volume_dict


def run_pipeline(
    input_path: str | Path,
    out_dir: str | Path,
    config_path: str | Path,
    units: str = "m",
    segmentation: str | None = None,
    roi_json: str | None = None,
    known_scale_json: str | None = None,
    overwrite: bool = False,
    command: str = "",
) -> dict[str, Any]:
    config = load_config(config_path)
    pipeline = VolumePipeline(
        input_path=Path(input_path),
        out_dir=Path(out_dir),
        config=config,
        units=units,
        segmentation=segmentation,
        roi_json=roi_json,
        known_scale_json=known_scale_json,
        overwrite=overwrite,
        command=command,
    )
    return pipeline.run()

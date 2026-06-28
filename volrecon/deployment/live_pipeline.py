"""Live ZED capture + reconstruction pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from rich.console import Console

from volrecon.camera.capture_session import CaptureSession, CaptureSessionConfig
from volrecon.camera.zed_capture import ZEDCaptureConfig, ZEDStereoCapture
from volrecon.config import PROJECT_ROOT
from volrecon.deployment.live_config import LivePipelineConfig
from volrecon.deployment.run_manager import run_paths, save_environment
from volrecon.fusion.bounds import compute_bounds_from_depth_points, robust_expand_bounds
from volrecon.fusion.fusion_utils import world_cam_from_view, PoseConventionError
from volrecon.fusion.open3d_tsdf import PlainTSDFConfig
from volrecon.fusion.plain_tsdf import PlainTSDFRunConfig, run_plain_tsdf, group_views_by_scene
from volrecon.fusion.weighted_tsdf import DenseChunkedWeightedTSDF, WeightedTSDFConfig
from volrecon.fusion.weighted_volume import compute_weighted_volumes
from volrecon.geometry.mesh_volume import compute_mesh_volume_report
from volrecon.io.json_io import read_jsonl, write_json
from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.stereo.foundation_stereo_wrapper import FoundationStereoConfig, FoundationStereoWrapper, resolve_view_paths
from volrecon.uncertainty.calibration import UncertaintyConfig
from volrecon.uncertainty.uncertainty_model import compute_view_uncertainty
from volrecon.visualization.html_report import write_html_report
from volrecon.visualization.zed_scene_viz import write_zed_scene_visualizations

logger = logging.getLogger(__name__)
console = Console()


class LiveReconstructionPipeline:
    def __init__(self, config: LivePipelineConfig) -> None:
        self.config = config

    def _scene_dir(self) -> Path:
        return (self.config.output_root / self.config.scene_name).resolve()

    def _manifest_path(self) -> Path:
        return self._scene_dir() / "manifest.jsonl"

    def run_capture_only(self) -> Path:
        cfg = self.config
        if cfg.dry_run:
            capture = ZEDStereoCapture(cfg.zed)
            capture.open()
            calib = capture.get_calibration()
            capture.close()
            console.print("[green]Dry-run OK[/green] baseline=%.4fm" % calib["baseline_m"])
            return self._scene_dir()

        capture = ZEDStereoCapture(cfg.zed)
        capture.open()
        try:
            session = CaptureSession(
                capture,
                CaptureSessionConfig(
                    scene_name=cfg.scene_name,
                    output_root=cfg.output_root,
                    num_keyframes=cfg.capture_num_keyframes,
                    pose_mode=cfg.pose_mode,
                    overwrite=cfg.overwrite,
                    save_preview_video=cfg.zed.save_preview_video,
                ),
                cfg.zed,
            )
            scene_dir = session.capture_keyframes(cfg.capture_num_keyframes)
            save_environment(scene_dir, {"mode": "capture_only"})
            console.print(f"[bold green]Captured[/bold green] {scene_dir}")
            self._print_next_commands(scene_dir)
            return scene_dir
        finally:
            capture.close()

    def _run_foundation_stereo(self, scene_dir: Path) -> Path:
        paths = run_paths(scene_dir, self.config.fusion.method.replace("_tsdf", "_tsdf"))
        depth_out = scene_dir / "runs" / "plain_tsdf" / "depth_predictions"
        if self.config.fusion.method == "weighted_tsdf":
            depth_out = scene_dir / "runs" / "weighted_tsdf" / "depth_predictions"
        depth_out.mkdir(parents=True, exist_ok=True)

        fs_cfg = FoundationStereoConfig(
            foundationstereo_repo=self.config.stereo_depth.foundationstereo_repo,
            ckpt=self.config.stereo_depth.checkpoint,
            scale=self.config.stereo_depth.scale,
            valid_iters=self.config.stereo_depth.valid_iters,
            min_depth_m=self.config.stereo_depth.min_depth_m,
            max_depth_m=self.config.stereo_depth.max_depth_m,
        )
        wrapper = FoundationStereoWrapper(fs_cfg)
        records = [ViewRecord.from_dict(r) for r in read_jsonl(self._manifest_path())]

        valid_ratios = []
        for view in records:
            out_dir = depth_out / view.scene_id / view.view_id
            if (out_dir / "depth_m.npy").exists() and not self.config.overwrite_depth:
                continue
            left, right = resolve_view_paths(view, PROJECT_ROOT)
            pred = wrapper.run_view(view, left, right, out_dir)
            valid_ratios.append(float(pred.valid_mask.mean()))

        save_environment(scene_dir, {"foundationstereo_repo": str(fs_cfg.foundationstereo_repo)})
        console.print(f"Depth predictions: {depth_out} mean_valid={np.mean(valid_ratios) if valid_ratios else 0:.2%}")
        return depth_out

    def _run_tsdf(self, scene_dir: Path, depth_out: Path, weighted: bool = False) -> Path:
        manifest = self._manifest_path()
        records = [ViewRecord.from_dict(r) for r in read_jsonl(manifest)]
        by_scene = group_views_by_scene(manifest)

        if weighted:
            recon_base = scene_dir / "runs" / "weighted_tsdf" / "reconstructions"
            unc_base = scene_dir / "runs" / "weighted_tsdf" / "uncertainty"
            unc_cfg = UncertaintyConfig()
            for scene_id, views in by_scene.items():
                for view in views:
                    pd = depth_out / scene_id / view.view_id
                    ud = unc_base / scene_id / view.view_id
                    if not (ud / "weight_total.npy").exists():
                        compute_view_uncertainty(view, pd, ud, unc_cfg, scene_views=views, depth_pred_root=depth_out)

            wcfg = WeightedTSDFConfig(
                voxel_length_m=self.config.fusion.voxel_length_m,
                sdf_trunc_m=self.config.fusion.sdf_trunc_m,
                max_depth_m=self.config.fusion.depth_trunc_m,
            )
            for scene_id, views in by_scene.items():
                self._reconstruct_weighted_scene(scene_id, views, depth_out, unc_base, recon_base / scene_id, wcfg)
            return recon_base

        run_cfg = PlainTSDFRunConfig(
            manifest_path=manifest,
            depth_predictions_root=depth_out,
            out_root=scene_dir / "runs" / "plain_tsdf" / "reconstructions",
            tsdf=PlainTSDFConfig(
                voxel_length_m=self.config.fusion.voxel_length_m,
                sdf_trunc_m=self.config.fusion.sdf_trunc_m,
                depth_trunc_m=self.config.fusion.depth_trunc_m,
            ),
        )
        from volrecon.fusion.plain_tsdf import run_plain_tsdf

        run_plain_tsdf(run_cfg)
        return run_cfg.out_root

    def _write_scene_visualizations(self, scene_dir: Path, recon_out: Path, weighted: bool) -> None:
        try:
            write_zed_scene_visualizations(scene_dir, recon_out, weighted=weighted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Visualization failed: %s", exc)

    def _reconstruct_weighted_scene(self, scene_id, views, depth_out, unc_base, out_dir, wcfg):
        depth_maps, Ks, T_wcs, frames = [], [], [], []
        for view in views:
            pd = depth_out / scene_id / view.view_id
            ud = unc_base / scene_id / view.view_id
            if not (pd / "depth_m.npy").exists():
                continue
            try:
                T_wc, T_cw = world_cam_from_view(view)
            except PoseConventionError:
                if not self.config.allow_no_pose_single_view:
                    raise
                continue
            depth_maps.append(np.load(pd / "depth_m.npy"))
            Ks.append(view.K)
            T_wcs.append(T_wc)
            frames.append((view, np.load(pd / "depth_m.npy"), np.load(ud / "weight_total.npy"), view.K, T_cw))

        bounds = robust_expand_bounds(compute_bounds_from_depth_points(depth_maps, Ks, T_wcs), 0.05)
        tsdf = DenseChunkedWeightedTSDF(bounds, wcfg)
        for view, depth, weight, K, T_cw in frames:
            tsdf.integrate_view(depth, weight, K, T_cw, view_id=view.view_id)
        paths = tsdf.save_outputs(out_dir)
        vol = compute_weighted_volumes(paths["mesh_weighted_clean"], wcfg.voxel_length_m)
        write_json(out_dir / "volume.json", vol.to_dict())

    def run_capture_then_reconstruct(self) -> Path:
        scene_dir = self.run_capture_only() if not self._manifest_path().exists() else self._scene_dir()
        if self.config.dry_run:
            return scene_dir
        depth_out = self._run_foundation_stereo(scene_dir)
        weighted = self.config.fusion.method == "weighted_tsdf"
        recon_out = self._run_tsdf(scene_dir, depth_out, weighted=weighted)
        self._write_scene_visualizations(scene_dir, recon_out, weighted)
        self._print_summary(scene_dir, recon_out, weighted)
        return scene_dir

    def run_live_incremental_plain_tsdf(self) -> Path:
        return self.run_capture_then_reconstruct()

    def run_live_incremental_weighted_tsdf(self) -> Path:
        self.config.fusion.method = "weighted_tsdf"
        return self.run_capture_then_reconstruct()

    def _print_summary(self, scene_dir: Path, recon_out: Path, weighted: bool) -> None:
        mesh_name = "mesh_weighted_clean.ply" if weighted else "mesh_clean.ply"
        scenes = list(recon_out.iterdir()) if recon_out.exists() else []
        n_views = len(list(read_jsonl(self._manifest_path())))
        vol_report = None
        for sd in scenes:
            if not sd.is_dir():
                continue
            mesh = sd / mesh_name
            if mesh.exists():
                rep = compute_mesh_volume_report(__import__("trimesh").load(mesh, force="mesh"))
                vol_report = {
                    "volume_m3": rep.volume_m3,
                    "volume_liters": rep.volume_liters,
                    "mesh_watertight": rep.mesh_watertight,
                }
                console.print(f"Volume: {rep.volume_m3:.4f} m³ ({rep.volume_liters:.2f} L) watertight={rep.mesh_watertight}")
        write_html_report(
            scene_dir / "report.html",
            scene_dir.name,
            metrics={"num_views": n_views, "method": "weighted_tsdf" if weighted else "plain_tsdf", "dataset": "zed_live"},
            volume=vol_report or {"note": "no mesh volume computed"},
            image_paths={
                "left_right_grid": scene_dir / "left_right_grid.png",
                "camera_trajectory": scene_dir / "camera_trajectory.png",
                "depth_grid": scene_dir / "runs" / ("weighted_tsdf" if weighted else "plain_tsdf") / "depth_predictions" / "depth_grid.png",
                "mesh_preview": scene_dir / "mesh_preview.png",
            },
        )
        console.print(f"Report: {scene_dir / 'report.html'}")
        console.print("[yellow]Warnings:[/yellow] no GT for live ZED; volume is visible-scene estimate only.")

    def _print_next_commands(self, scene_dir: Path) -> None:
        manifest = scene_dir / "manifest.jsonl"
        console.print(f"\nNext: python -m volrecon.scripts.zed_run_capture_then_reconstruct --scene_name {scene_dir.name} ...")
        console.print(f"Manifest: {manifest}")

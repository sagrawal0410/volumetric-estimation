"""FoundationStereo integration wrapper (subprocess + optional Python API)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from volrecon.config import PROJECT_ROOT
from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.datasets.preprocessing import foundation_stereo_usable
from volrecon.geometry.camera import depth_to_pointcloud, disparity_to_depth, resize_intrinsics
from volrecon.io.image_io import read_rgb, write_image
from volrecon.io.json_io import write_json
from volrecon.stereo.confidence import apply_depth_mask, build_valid_depth_mask
from volrecon.stereo.types import DepthPrediction

logger = logging.getLogger(__name__)

NO_STEREO_ERROR = (
    "No true stereo pair available for this record. "
    "Use ROBI true stereo, external stereo, or BOP synthetic_stereo_from_bop_mesh mode."
)


@dataclass
class FoundationStereoConfig:
    foundationstereo_repo: Path
    ckpt: Path
    scale: float = 1.0
    valid_iters: int = 16
    min_disp: float = 0.5
    min_depth_m: float = 0.1
    max_depth_m: float = 2.0
    hiera: int = 0
    z_far: float = 100.0
    get_pc: bool = True
    remove_invisible: bool = True
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)


class FoundationStereoWrapper:
    def __init__(self, cfg: FoundationStereoConfig) -> None:
        self.cfg = cfg
        self._api_available = self._probe_python_api()

    def _probe_python_api(self) -> bool:
        repo = self.cfg.foundationstereo_repo
        if not repo.exists():
            return False
        try:
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            import core.foundation_stereo  # noqa: F401, WPS433

            return True
        except Exception:  # noqa: BLE001
            return False

    def validate_view(self, view: ViewRecord) -> None:
        if not foundation_stereo_usable(view):
            raise ValueError(NO_STEREO_ERROR)
        if view.stereo is None or view.stereo.baseline_m is None:
            if view.stereo and view.stereo.synthetic and view.stereo.baseline_m:
                return
            raise ValueError(f"Missing stereo baseline for {view.scene_id}/{view.view_id}")
        if view.K is None:
            raise ValueError(f"Missing intrinsics K for {view.scene_id}/{view.view_id}")
        if view.stereo and not view.stereo.rectified:
            logger.warning(
                "View %s/%s: stereo marked not rectified; FoundationStereo expects rectified pairs.",
                view.scene_id,
                view.view_id,
            )

    def run_view(
        self,
        view: ViewRecord,
        left_path: Path,
        right_path: Path,
        out_dir: Path,
    ) -> DepthPrediction:
        self.validate_view(view)
        if view.gt_depth_path:
            logger.debug("GT depth path present but ignored for inference: %s", view.gt_depth_path)

        out_dir.mkdir(parents=True, exist_ok=True)
        left = read_rgb(left_path)
        right = read_rgb(right_path)
        K = np.asarray(view.K, dtype=np.float64).reshape(3, 3)
        baseline_m = float(view.stereo.baseline_m)  # type: ignore[union-attr]

        scale = self.cfg.scale
        if scale != 1.0:
            new_w = int(left.shape[1] * scale)
            new_h = int(left.shape[0] * scale)
            left = cv2.resize(left, (new_w, new_h), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (new_w, new_h), interpolation=cv2.INTER_AREA)
            K = resize_intrinsics(K, scale, scale)

        work_left = out_dir / "left_scaled.png"
        work_right = out_dir / "right_scaled.png"
        write_image(work_left, left)
        write_image(work_right, right)
        write_json(out_dir / "K_scaled.json", {"K": K.tolist(), "scale": scale})

        disparity = self._run_inference(work_left, work_right, out_dir)
        depth_m = disparity_to_depth(disparity, fx_px=float(K[0, 0]), baseline_m=baseline_m)
        if isinstance(depth_m, float):
            depth_m = np.full(disparity.shape, depth_m, dtype=np.float64)
        depth_m = np.asarray(depth_m, dtype=np.float64)

        valid_mask = build_valid_depth_mask(
            disparity,
            depth_m,
            min_disp=self.cfg.min_disp,
            min_depth_m=self.cfg.min_depth_m,
            max_depth_m=self.cfg.max_depth_m,
        )
        depth_m = apply_depth_mask(depth_m, valid_mask)

        np.save(out_dir / "disparity.npy", disparity.astype(np.float32))
        np.save(out_dir / "depth_m.npy", depth_m.astype(np.float32))
        cv2.imwrite(str(out_dir / "valid_mask.png"), (valid_mask.astype(np.uint8) * 255))

        depth_vis = self._depth_colormap(depth_m, valid_mask)
        cv2.imwrite(str(out_dir / "depth_colormap.png"), depth_vis)

        rgb_for_pc = left
        points, colors = depth_to_pointcloud(depth_m, K, rgb=rgb_for_pc, mask=valid_mask)
        self._save_pointcloud_ply(out_dir / "pointcloud_est.ply", points, colors)

        meta = {
            "scene_id": view.scene_id,
            "view_id": view.view_id,
            "baseline_m": baseline_m,
            "K": K.tolist(),
            "scale": scale,
            "min_depth_m": self.cfg.min_depth_m,
            "max_depth_m": self.cfg.max_depth_m,
            "valid_pixel_ratio": float(valid_mask.mean()),
            "backend": "python_api" if self._api_available else "subprocess",
            "synthetic": view.synthetic,
        }
        write_json(out_dir / "stereo_debug.json", meta)

        return DepthPrediction(
            disparity_px=disparity,
            depth_m=depth_m,
            valid_mask=valid_mask,
            K=K,
            baseline_m=baseline_m,
            meta=meta,
        )

    def _run_inference(self, left_path: Path, right_path: Path, out_dir: Path) -> np.ndarray:
        disp_out = out_dir / "disparity_raw.npy"
        if self._api_available:
            try:
                return self._run_python_api(left_path, right_path, out_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FoundationStereo Python API failed (%s); falling back to subprocess.", exc)
        return self._run_subprocess(left_path, right_path, out_dir, disp_out)

    def _run_subprocess(self, left_path: Path, right_path: Path, out_dir: Path, disp_out: Path) -> np.ndarray:
        repo = self.cfg.foundationstereo_repo.resolve()
        ckpt = self.cfg.ckpt.resolve()
        demo = repo / "scripts" / "run_demo.py"
        if not demo.exists():
            demo = repo / "run_demo.py"
        if not demo.exists():
            raise FileNotFoundError(f"FoundationStereo run_demo.py not found under {repo}")

        cmd = [
            sys.executable,
            str(demo),
            "--left_file",
            str(left_path),
            "--right_file",
            str(right_path),
            "--ckpt_dir",
            str(ckpt.parent if ckpt.suffix == ".pth" else ckpt),
            "--out_dir",
            str(out_dir),
            "--valid_iters",
            str(self.cfg.valid_iters),
            "--scale",
            str(self.cfg.scale),
            "--hiera",
            str(self.cfg.hiera),
            "--z_far",
            str(self.cfg.z_far),
        ]
        if ckpt.suffix == ".pth":
            cmd.extend(["--ckpt", str(ckpt)])

        logger.info("Running FoundationStereo subprocess: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=str(repo), check=True)

        for candidate in [
            out_dir / "disparity.npy",
            out_dir / "disp.npy",
            out_dir / "output" / "disparity.npy",
        ]:
            if candidate.exists():
                return np.load(candidate).astype(np.float64)

        vis = sorted(out_dir.glob("*disp*.npy"))
        if vis:
            return np.load(vis[0]).astype(np.float64)
        raise FileNotFoundError(f"Disparity output not found in {out_dir}")

    def _run_python_api(self, left_path: Path, right_path: Path, out_dir: Path) -> np.ndarray:
        repo = str(self.cfg.foundationstereo_repo.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

        import torch  # noqa: WPS433
        from core.foundation_stereo import FoundationStereo  # noqa: WPS433
        from core.utils.utils import InputPadder  # noqa: WPS433

        ckpt = self.cfg.ckpt
        model = FoundationStereo()
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval()

        left = read_rgb(left_path)
        right = read_rgb(right_path)
        img0 = torch.from_numpy(left).permute(2, 0, 1).float()[None]
        img1 = torch.from_numpy(right).permute(2, 0, 1).float()[None]
        padder = InputPadder(img0.shape, divis_by=32)
        img0, img1 = padder.pad(img0, img1)

        with torch.no_grad():
            disp = model(img0, img1, iters=self.cfg.valid_iters, test_mode=True)
        disp = padder.unpad(disp)[0, 0].cpu().numpy().astype(np.float64)
        np.save(out_dir / "disparity_raw.npy", disp.astype(np.float32))
        return disp

    @staticmethod
    def _depth_colormap(depth_m: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        vis = np.zeros(depth_m.shape, dtype=np.float64)
        if np.any(valid_mask):
            d = depth_m[valid_mask]
            vis[valid_mask] = (d - d.min()) / max(d.max() - d.min(), 1e-6)
        cm = (vis * 255).astype(np.uint8)
        return cv2.applyColorMap(cm, cv2.COLORMAP_TURBO)

    @staticmethod
    def _save_pointcloud_ply(path: Path, points: np.ndarray, colors: np.ndarray | None) -> None:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
        path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(path), pcd)


def resolve_view_paths(view: ViewRecord, project_root: Path) -> tuple[Path, Path]:
    if view.left_path is None or view.right_path is None:
        raise ValueError(NO_STEREO_ERROR)

    def resolve(p: Path | str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (project_root / path).resolve()

    return resolve(view.left_path), resolve(view.right_path)

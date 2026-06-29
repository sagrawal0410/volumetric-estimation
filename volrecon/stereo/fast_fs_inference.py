"""Headless Fast-FoundationStereo inference for volrecon (no GUI / cv2.imshow)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def _prepare_repo(repo: Path) -> None:
    repo = repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"Fast-FoundationStereo repo not found: {repo}")
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def run_fast_fs_inference(
    repo: Path,
    model_path: Path,
    left_path: Path,
    right_path: Path,
    out_dir: Path,
    *,
    valid_iters: int = 8,
    max_disp: int = 192,
    scale: float = 1.0,
    hiera: int = 0,
) -> np.ndarray:
    """
    Run Fast-FoundationStereo forward pass and save ``disparity.npy`` under ``out_dir``.

    Images are resized when ``scale != 1``. When volrecon has already scaled inputs,
    pass ``scale=1.0``.
    """
    from volrecon.stereo.stereo_backends import require_cfg_yaml

    model_path = model_path.resolve()
    require_cfg_yaml(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    _prepare_repo(repo)
    from core.utils.utils import InputPadder  # noqa: WPS433
    from Utils import AMP_DTYPE  # noqa: WPS433

    with (model_path.parent / "cfg.yaml").open("r", encoding="utf-8") as ff:
        cfg: dict = yaml.safe_load(ff)
    cfg["valid_iters"] = valid_iters
    cfg["max_disp"] = max_disp
    cfg["scale"] = scale
    cfg["hiera"] = hiera

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Fast-FoundationStereo inference requires CUDA. "
            "On Jetson, install PyTorch from https://pypi.jetson-ai-lab.io/jp6/cu126"
        )

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.cuda().eval()

    img0 = imageio.imread(left_path)
    img1 = imageio.imread(right_path)
    if img0.ndim == 2:
        img0 = np.tile(img0[..., None], (1, 1, 3))
    if img1.ndim == 2:
        img1 = np.tile(img1[..., None], (1, 1, 3))
    img0 = img0[..., :3]
    img1 = img1[..., :3]

    if scale != 1.0:
        img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
        img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
    h, w = img0.shape[:2]

    img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        if not hiera:
            disp = model.forward(
                img0_t,
                img1_t,
                iters=valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
        else:
            disp = model.run_hierachical(
                img0_t,
                img1_t,
                iters=valid_iters,
                test_mode=True,
                small_ratio=0.5,
            )

    disp = padder.unpad(disp.float())
    disp_np = disp.data.cpu().numpy().reshape(h, w).clip(0, None).astype(np.float64)

    np.save(out_dir / "disparity.npy", disp_np.astype(np.float32))
    np.save(out_dir / "disparity_raw.npy", disp_np.astype(np.float32))
    logger.info("Fast-FS disparity saved: %s shape=%s", out_dir / "disparity.npy", disp_np.shape)
    return disp_np


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless Fast-FoundationStereo single-view inference.")
    parser.add_argument("--repo", required=True, type=Path, help="Fast-FoundationStereo clone root")
    parser.add_argument("--model_dir", required=True, type=Path, help="Path to model_best_bp2_serialize.pth")
    parser.add_argument("--left_file", required=True, type=Path)
    parser.add_argument("--right_file", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--valid_iters", type=int, default=8)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--hiera", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_fast_fs_inference(
        args.repo,
        args.model_dir,
        args.left_file,
        args.right_file,
        args.out_dir,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
        scale=args.scale,
        hiera=args.hiera,
    )


if __name__ == "__main__":
    main()

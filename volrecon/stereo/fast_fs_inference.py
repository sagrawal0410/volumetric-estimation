"""Headless Fast-FoundationStereo inference for volrecon (no GUI / cv2.imshow)."""

from __future__ import annotations

import gc
import os
import platform

# Jetson/aarch64 PyTorch builds ship without a working Triton stack; disable dynamo early.
if platform.machine().lower() in {"aarch64", "arm64"}:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import json
import logging
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

_FAST_FS_COMPILED_KERNELS = (
    "build_gwc_volume_optimized_pytorch1",
    "build_concat_volume_optimized_pytorch1",
    "build_concat_volume_optimized_pytorch",
)

_JETSON_DEFAULT_SCALE = 0.5


def _embedded_gpu() -> bool:
    return platform.machine().lower() in {"aarch64", "arm64"}


def _triton_usable() -> bool:
    """True only when Triton imports and exposes a JIT compiler (desktop CUDA builds)."""
    import importlib.util

    if importlib.util.find_spec("triton") is None:
        return False
    try:
        import triton  # noqa: F401, WPS433

        return callable(getattr(triton, "jit", None))
    except Exception:  # noqa: BLE001
        return False


def _unwrap_torch_compiled(fn):
    seen: set[int] = set()
    while id(fn) not in seen:
        seen.add(id(fn))
        for attr in ("_orig_mod", "__wrapped__", "_torchdynamo_orig_callable"):
            nxt = getattr(fn, attr, None)
            if nxt is not None and callable(nxt):
                fn = nxt
                break
        else:
            break
    return fn


def _disable_torch_compile(*, force: bool) -> None:
    """
    Disable ``torch.compile`` for Fast-FoundationStereo inference.

    Fast-FS decorates GWC/concat volume builders with ``@torch.compile``. Without a
    working Triton install (typical on Jetson), inductor fails on the first forward pass.
    """
    if not force and _triton_usable():
        return

    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    try:
        import torch._dynamo as dynamo

        dynamo.config.disable = True
        dynamo.config.suppress_errors = True
    except Exception:  # noqa: BLE001
        pass

    def _compile_identity(fn, *_args, **_kwargs):
        return fn

    torch.compile = _compile_identity  # type: ignore[misc, assignment]
    logger.info("Using eager Fast-FoundationStereo kernels (torch.compile disabled).")


def _patch_fast_fs_compiled_kernels() -> None:
    """Replace already-wrapped compiled kernels with their eager implementations."""
    import core.foundation_stereo as fs  # noqa: WPS433
    import core.submodule as sm  # noqa: WPS433

    for mod in (sm, fs):
        for name in _FAST_FS_COMPILED_KERNELS:
            if hasattr(mod, name):
                setattr(mod, name, _unwrap_torch_compiled(getattr(mod, name)))


def _set_model_arg(model, name: str, value) -> None:
    args = getattr(model, "args", None)
    if args is None:
        return
    if isinstance(args, dict):
        args[name] = value
    elif hasattr(args, "__setitem__"):
        try:
            args[name] = value
        except Exception:  # noqa: BLE001
            setattr(args, name, value)
    else:
        setattr(args, name, value)


def _resolve_inference_scale(
    scale: float,
    *,
    force_full_resolution: bool,
    pre_scaled: bool,
) -> float:
    if pre_scaled:
        return 1.0
    if force_full_resolution or not _embedded_gpu() or scale < 1.0:
        return scale
    if scale >= 1.0:
        logger.warning(
            "Jetson/embedded GPU detected: overriding --scale %.2f -> %.1f to avoid OOM. "
            "Pass --force-full-resolution to keep full resolution (may be killed by the OOM killer).",
            scale,
            _JETSON_DEFAULT_SCALE,
        )
        return _JETSON_DEFAULT_SCALE
    return scale


def _load_rgb_pair(left_path: Path, right_path: Path) -> tuple[np.ndarray, np.ndarray]:
    img0 = imageio.imread(left_path)
    img1 = imageio.imread(right_path)
    if img0.ndim == 2:
        img0 = np.tile(img0[..., None], (1, 1, 3))
    if img1.ndim == 2:
        img1 = np.tile(img1[..., None], (1, 1, 3))
    return img0[..., :3], img1[..., :3]


def _resize_stereo_pair(img0: np.ndarray, img1: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    if scale == 1.0:
        return img0, img1
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
    return img0, img1


def _load_fast_fs_model(model_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Fast-FoundationStereo inference requires CUDA. "
            "On Jetson, install PyTorch from https://pypi.jetson-ai-lab.io/jp6/cu126"
        )
    map_location = "cuda" if _embedded_gpu() else "cpu"
    logger.info("Loading Fast-FS checkpoint: %s (map_location=%s)", model_path, map_location)
    model = torch.load(model_path, map_location=map_location, weights_only=False)
    if map_location == "cpu":
        model = model.cuda()
    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


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
    low_memory: bool | None = None,
    allow_torch_compile: bool = False,
    force_full_resolution: bool = False,
    pre_scaled: bool = False,
) -> np.ndarray:
    """
    Run Fast-FoundationStereo forward pass and save ``disparity.npy`` under ``out_dir``.

    Images are resized when ``scale != 1``. When volrecon has already scaled inputs,
    pass ``pre_scaled=True`` (and ``scale=1.0``).
    """
    from volrecon.stereo.stereo_backends import prepare_stereo_repo, require_cfg_yaml, resolve_repo_path

    if low_memory is None:
        low_memory = _embedded_gpu()

    _disable_torch_compile(force=not allow_torch_compile)

    repo = resolve_repo_path(repo)
    model_path = model_path.expanduser().resolve()
    left_path = left_path.expanduser().resolve()
    right_path = right_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    require_cfg_yaml(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = _resolve_inference_scale(
        scale,
        force_full_resolution=force_full_resolution,
        pre_scaled=pre_scaled,
    )

    prepare_stereo_repo(repo, backend="fast_foundation_stereo")
    import core.submodule  # noqa: F401, WPS433

    _patch_fast_fs_compiled_kernels()
    from core.utils.utils import InputPadder  # noqa: WPS433
    from Utils import AMP_DTYPE  # noqa: WPS433

    img0, img1 = _load_rgb_pair(left_path, right_path)
    img0, img1 = _resize_stereo_pair(img0, img1, scale)
    h, w = img0.shape[:2]
    logger.info(
        "Fast-FS inference: shape=%dx%d scale=%.2f valid_iters=%d max_disp=%d low_memory=%s",
        w,
        h,
        scale,
        valid_iters,
        max_disp,
        low_memory,
    )

    model = _load_fast_fs_model(model_path)
    _set_model_arg(model, "valid_iters", valid_iters)
    _set_model_arg(model, "max_disp", max_disp)
    _set_model_arg(model, "low_memory", low_memory)

    img0_t = torch.as_tensor(img0, device="cuda").float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1, device="cuda").float()[None].permute(0, 3, 1, 2)
    del img0, img1
    gc.collect()

    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        if not hiera:
            disp = model.forward(
                img0_t,
                img1_t,
                iters=valid_iters,
                test_mode=True,
                low_memory=low_memory,
                optimize_build_volume="pytorch1",
            )
        else:
            disp = model.run_hierachical(
                img0_t,
                img1_t,
                iters=valid_iters,
                test_mode=True,
                low_memory=low_memory,
                small_ratio=0.5,
            )

    disp = padder.unpad(disp.float())
    disp_np = disp.data.cpu().numpy().reshape(h, w).clip(0, None).astype(np.float64)

    np.save(out_dir / "disparity.npy", disp_np.astype(np.float32))
    np.save(out_dir / "disparity_raw.npy", disp_np.astype(np.float32))
    meta = {
        "scale": scale,
        "pre_scaled": pre_scaled,
        "valid_iters": valid_iters,
        "max_disp": max_disp,
        "low_memory": low_memory,
        "shape": [h, w],
    }
    (out_dir / "fast_fs_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    logger.info("Fast-FS disparity saved: %s shape=%s", out_dir / "disparity.npy", disp_np.shape)
    return disp_np


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless Fast-FoundationStereo single-view inference.")
    parser.add_argument("--repo", required=True, type=Path, help="Fast-FoundationStereo clone root")
    parser.add_argument("--model_dir", required=True, type=Path, help="Path to model_best_bp2_serialize.pth")
    parser.add_argument("--left_file", required=True, type=Path)
    parser.add_argument("--right_file", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--valid_iters", type=int, default=4 if _embedded_gpu() else 8)
    parser.add_argument("--max_disp", type=int, default=192)
    default_scale = _JETSON_DEFAULT_SCALE if _embedded_gpu() else 1.0
    parser.add_argument(
        "--scale",
        type=float,
        default=default_scale,
        help=f"Image resize factor before inference (default {default_scale} on Jetson).",
    )
    parser.add_argument("--hiera", type=int, default=0)
    parser.add_argument(
        "--low-memory",
        dest="low_memory",
        action="store_true",
        default=_embedded_gpu(),
        help="Use Fast-FS low_memory forward path (default on Jetson).",
    )
    parser.add_argument(
        "--no-low-memory",
        dest="low_memory",
        action="store_false",
        help="Disable Fast-FS low_memory path.",
    )
    parser.add_argument(
        "--pre-scaled",
        action="store_true",
        help="Input images are already resized (skip all scaling; used by volrecon wrapper).",
    )
    parser.add_argument(
        "--force-full-resolution",
        action="store_true",
        help="Do not auto-downscale on Jetson (may OOM).",
    )
    parser.add_argument(
        "--allow-torch-compile",
        action="store_true",
        help="Allow torch.compile (requires working Triton; not supported on Jetson).",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_fast_fs_inference(
        args.repo,
        args.model_dir.expanduser().resolve(),
        args.left_file.expanduser().resolve(),
        args.right_file.expanduser().resolve(),
        args.out_dir.expanduser().resolve(),
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
        scale=args.scale,
        hiera=args.hiera,
        low_memory=args.low_memory,
        allow_torch_compile=args.allow_torch_compile and not args.eager,
        force_full_resolution=args.force_full_resolution,
        pre_scaled=args.pre_scaled,
    )


if __name__ == "__main__":
    main()

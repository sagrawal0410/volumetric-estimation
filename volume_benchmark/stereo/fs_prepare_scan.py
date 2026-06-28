"""Convert prepared stereo scans to RGB-D scans using FoundationStereo depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from volume_benchmark.common.io import Frame, save_prepared_scan
from volume_benchmark.stereo.disparity_depth import (
    disparity_to_depth_m,
    save_depth_debug,
    save_disparity_debug,
)
from volume_benchmark.stereo.foundation_stereo_backend import FoundationStereoBackend
from volume_benchmark.stereo.rectification import validate_rectified_pair
from volume_benchmark.stereo.stereo_dataset_adapter import load_prepared_stereo_scan


def fs_stereo_to_rgbd_scan(
    stereo_scan_dir: str | Path,
    out_scan_dir: str | Path,
    backend: FoundationStereoBackend,
    max_depth_m: float = 5.0,
    min_disp: float = 0.1,
    save_debug: bool = True,
) -> Path:
    """Run FoundationStereo on each frame and write normalized RGB-D scan."""
    stereo = load_prepared_stereo_scan(stereo_scan_dir)
    out = Path(out_scan_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    fx = float(stereo.K_left[0, 0])
    baseline = float(stereo.baseline_m)

    rgbd_frames: list[Frame] = []
    frame_meta: dict[str, dict] = {}

    for idx, sf in enumerate(stereo.frames):
        validate_rectified_pair(sf.left_rgb, sf.right_rgb)
        disparity = backend.predict_disparity(sf.left_rgb, sf.right_rgb)
        depth_m = disparity_to_depth_m(disparity, fx, baseline, min_disp=min_disp, max_depth_m=max_depth_m)
        depth_m[~sf.mask] = 0.0

        prov: dict[str, Any] = {
            "depth_source": "foundationstereo",
            "stereo_source": stereo.metadata.get("source_mode", sf.meta.get("source_mode", "unknown")),
            "fx_px": fx,
            "baseline_m": baseline,
            "checkpoint_path": str(backend.checkpoint_path),
            "model_variant": backend.variant,
            "frame_index": idx,
        }
        prov.update(sf.meta)

        rgbd_frames.append(
            Frame(
                depth_m=depth_m.astype(np.float32),
                mask=sf.mask.astype(bool),
                T_cam_to_object=sf.T_left_cam_to_object.astype(np.float64),
                source_info=prov,
            )
        )
        frame_meta[str(idx)] = prov

        if save_debug:
            dbg = out / "debug" / f"frame_{idx:03d}"
            dbg.mkdir(parents=True, exist_ok=True)
            np.save(dbg / "disparity.npy", disparity)
            save_disparity_debug(disparity, dbg / "disparity_color.png")
            save_depth_debug(depth_m, dbg / "depth_debug.png")

    meta = dict(stereo.metadata)
    meta.update(
        {
            "depth_backend": "foundationstereo",
            "stereo_scan_dir": str(stereo.scan_dir),
            "baseline_m": baseline,
            "checkpoint_path": str(backend.checkpoint_path),
            "model_variant": backend.variant,
            "frame_source_info": frame_meta,
        }
    )

    import shutil

    gt_dst = out / "gt_mesh.ply"
    if stereo.gt_mesh_path.resolve() != gt_dst.resolve():
        shutil.copy2(stereo.gt_mesh_path, gt_dst)
    with (out / "gt_volume.json").open("w", encoding="utf-8") as f:
        json.dump(stereo.gt_volume, f, indent=2)
    save_prepared_scan(out, stereo.K_left, rgbd_frames, gt_dst, metadata=meta)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FoundationStereo depth scan preparation")
    parser.add_argument("--stereo_scan_dir", required=True)
    parser.add_argument("--out_scan_dir", required=True)
    parser.add_argument("--foundationstereo_repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", default="fast", choices=["fast", "full"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_depth_m", type=float, default=5.0)
    parser.add_argument("--max_input_size", type=int, nargs=2, default=None)
    args = parser.parse_args(argv)

    backend = FoundationStereoBackend(
        repo_path=args.foundationstereo_repo,
        checkpoint_path=args.checkpoint,
        variant=args.variant,
        device=args.device,
        max_input_size=tuple(args.max_input_size) if args.max_input_size else None,
    )
    out = fs_stereo_to_rgbd_scan(
        args.stereo_scan_dir,
        args.out_scan_dir,
        backend,
        max_depth_m=args.max_depth_m,
    )
    print(f"FoundationStereo RGB-D scan: {out}")


if __name__ == "__main__":
    main()

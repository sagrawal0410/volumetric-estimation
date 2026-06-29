"""Extract BOP T-LESS dataset to canonical processed format."""

from __future__ import annotations

import argparse
from pathlib import Path

from volrecon.config import DEFAULT_MANIFEST_DIR, DEFAULT_PROCESSED_ROOT, PreprocessConfig
from volrecon.datasets.bop import extract_bop_tless


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract BOP T-LESS to canonical processed format.")
    parser.add_argument("--root", required=True, type=Path, help="Path to T-LESS BOP root")
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument(
        "--mode",
        default="real_rgb_only",
        choices=["real_rgb_only", "synthetic_stereo_from_bop_mesh"],
    )
    parser.add_argument("--baseline_m", type=float, default=0.06)
    parser.add_argument("--out", default=DEFAULT_PROCESSED_ROOT / "bop_tless", type=Path)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_DIR / "bop_tless_manifest.jsonl", type=Path)
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_scenes", type=int, default=None, help="Process only first N scenes (low RAM)")
    parser.add_argument("--frame_stride", type=int, default=1, help="Keep every Nth frame")
    parser.add_argument("--max_views_per_scene", type=int, default=None, help="Cap views per scene")
    parser.add_argument(
        "--skip_union_voxels",
        action="store_true",
        help="Skip union_gt_voxels.npz (saves RAM on Jetson)",
    )
    args = parser.parse_args()

    cfg = PreprocessConfig(
        symlink=True if args.symlink or not args.copy else False,
        overwrite=args.overwrite,
        synthetic_baseline_m=args.baseline_m,
        processed_root=args.out.parent,
    )
    if args.copy:
        cfg.symlink = False

    extract_bop_tless(
        root=args.root,
        split=args.split,
        out_dir=args.out,
        manifest_path=args.manifest,
        cfg=cfg,
        mode=args.mode,
        baseline_m=args.baseline_m,
        max_scenes=args.max_scenes,
        frame_stride=args.frame_stride,
        max_views_per_scene=args.max_views_per_scene,
        skip_union_voxels=args.skip_union_voxels,
    )
    print(f"BOP T-LESS extraction complete. Manifest: {args.manifest}")


if __name__ == "__main__":
    main()

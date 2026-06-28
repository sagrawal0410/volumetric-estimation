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
    )
    print(f"BOP T-LESS extraction complete. Manifest: {args.manifest}")


if __name__ == "__main__":
    main()

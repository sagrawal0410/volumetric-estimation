"""Batch volume estimation."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from tqdm import tqdm

from rtabmap_volume.pipeline import run_pipeline

console = Console()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Batch volume estimation")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--pattern", default="*.ply")
    p.add_argument("--units", default="m")
    p.add_argument("--segmentation", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    input_dir = Path(args.input_dir)
    out_root = Path(args.out)
    files = sorted(input_dir.glob(args.pattern))

    if not files:
        console.print(f"[red]No files matching {args.pattern} in {input_dir}[/red]")
        return 1

    for f in tqdm(files, desc="Batch volume"):
        scene_out = out_root / f.stem
        run_pipeline(
            input_path=f,
            out_dir=scene_out,
            config_path=args.config,
            units=args.units,
            segmentation=args.segmentation,
            overwrite=args.overwrite,
            command=f"batch {f.name}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Primary CLI: estimate volume from RTAB-Map exports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from rtabmap_volume.io.rtabmap_export import try_export_from_db
from rtabmap_volume.pipeline import run_pipeline

console = Console()


def _default_config() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Estimate object/pile volume from RTAB-Map geometry exports")
    p.add_argument("--input", type=str, help="Path to mesh or point cloud (.ply, .pcd, .obj, ...)")
    p.add_argument("--out", type=str, required=True, help="Output run directory")
    p.add_argument("--config", type=str, default=str(_default_config()), help="YAML config path")
    p.add_argument("--units", type=str, default="m", choices=["m", "mm", "cm"], help="Input geometry units")
    p.add_argument("--mode", type=str, default="object_or_pile", help="Pipeline mode hint")
    p.add_argument("--segmentation", type=str, default=None, help="Segmentation mode override")
    p.add_argument("--roi_json", type=str, default=None, help="Manual ROI JSON path")
    p.add_argument("--interactive_crop", type=str, default="false", help="Use interactive crop (true/false)")
    p.add_argument("--known_scale_object_json", type=str, default=None, help="Known scale reference JSON")
    p.add_argument("--rtabmap_db", type=str, default=None, help="RTAB-Map database path")
    p.add_argument("--rtabmap_tools_path", type=str, default=None, help="Path to RTAB-Map CLI tools")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cmd = " ".join(sys.argv)

    input_path = args.input
    if args.rtabmap_db:
        out = Path(args.out)
        export = try_export_from_db(
            Path(args.rtabmap_db),
            out / "inputs",
            Path(args.rtabmap_tools_path) if args.rtabmap_tools_path else None,
        )
        if not export.success:
            console.print(f"[red]{export.message}[/red]")
            return 1
        input_path = str(export.exported_path)
        console.print(f"[green]Exported:[/green] {input_path}")
    elif not input_path:
        console.print("[red]Either --input or --rtabmap_db is required[/red]")
        return 1

    if args.interactive_crop.lower() == "true":
        from rtabmap_volume.scripts.crop_interactive import run_interactive_crop
        roi_path = Path(args.out) / "roi.json"
        run_interactive_crop(input_path, roi_path, args.units)
        args.roi_json = str(roi_path)
        args.segmentation = args.segmentation or "manual_aabb"

    run_pipeline(
        input_path=input_path,
        out_dir=args.out,
        config_path=args.config,
        units=args.units,
        segmentation=args.segmentation,
        roi_json=args.roi_json,
        known_scale_json=args.known_scale_object_json,
        overwrite=args.overwrite,
        command=cmd,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

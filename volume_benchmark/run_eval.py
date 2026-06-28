"""CLI for evaluating volume estimation methods on one prepared scan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from volume_benchmark.common.io import validate_prepared_scan
from volume_benchmark.methods.convex_hull import estimate_convex_hull_volume
from volume_benchmark.methods.tsdf import estimate_tsdf_volume
from volume_benchmark.methods.voxel_carving import estimate_voxel_carving_volume

METHODS = {
    "convex_hull": estimate_convex_hull_volume,
    "tsdf": estimate_tsdf_volume,
    "voxel_carving": estimate_voxel_carving_volume,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate volume estimation methods on a prepared scan."
    )
    parser.add_argument("scan_dir", type=Path, help="Prepared scan directory")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHODS),
        default=["convex_hull", "tsdf", "voxel_carving"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory (default: scan_dir/outputs)",
    )
    parser.add_argument("--voxel-downsample", type=float, default=0.002)
    parser.add_argument("--voxel-length", type=float, default=0.003)
    parser.add_argument("--voxel-size", type=float, default=0.004)
    parser.add_argument("--sdf-trunc", type=float, default=0.015)
    parser.add_argument("--depth-trunc", type=float, default=5.0)
    parser.add_argument("--depth-tolerance", type=float, default=0.010)
    parser.add_argument("--padding", type=float, default=0.02)
    parser.add_argument("--min-views-checked", type=int, default=2)
    parser.add_argument("--repair-mesh", action="store_true")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional combined JSON summary path",
    )
    parser.add_argument("--skip-validation", action="store_true")
    return parser


def _run_method(name: str, scan_dir: Path, args: argparse.Namespace) -> dict:
    fn = METHODS[name]
    if name == "convex_hull":
        return fn(
            scan_dir,
            output_dir=args.output_dir,
            voxel_downsample=args.voxel_downsample,
        )
    if name == "tsdf":
        return fn(
            scan_dir,
            output_dir=args.output_dir,
            voxel_length=args.voxel_length,
            sdf_trunc=args.sdf_trunc,
            depth_trunc=args.depth_trunc,
            repair_mesh=args.repair_mesh,
        )
    if name == "voxel_carving":
        return fn(
            scan_dir,
            output_dir=args.output_dir,
            voxel_size=args.voxel_size,
            depth_tolerance=args.depth_tolerance,
            padding=args.padding,
            min_views_checked=args.min_views_checked,
        )
    raise ValueError(f"Unknown method: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.skip_validation:
        errors = validate_prepared_scan(args.scan_dir)
        if errors:
            print(f"Invalid scan {args.scan_dir}:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1

    rows = []
    for method in args.methods:
        try:
            report = _run_method(method, args.scan_dir, args)
            rows.append(report)
            vol = report.get("volume_m3")
            gt = report.get("gt_volume_cm3")
            rel = report.get("relative_error_percent")
            if vol is None:
                print(f"{method}: volume unavailable — {report.get('warning', 'not watertight')}")
            else:
                print(
                    f"{method}: pred={vol * 1e6:.2f} cm³, gt={gt:.2f} cm³, rel_err={rel:.2f}%"
                )
        except Exception as exc:
            print(f"{method}: FAILED — {exc}", file=sys.stderr)
            rows.append({"method": method, "error": str(exc), "scan_dir": str(args.scan_dir)})

    summary = {"scan_dir": str(args.scan_dir), "results": rows}
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary to {args.summary}")

    failed = any("error" in r for r in rows)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

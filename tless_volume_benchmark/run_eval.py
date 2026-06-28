"""Run volume estimation methods on one prepared T-LESS scan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tless_volume_benchmark.methods.convex_hull import estimate_convex_hull
from tless_volume_benchmark.methods.tsdf import estimate_tsdf
from tless_volume_benchmark.methods.voxel_carving import estimate_voxel_carving
from tless_volume_benchmark.scan_io import load_prepared_scan

METHODS = {
    "convex_hull": estimate_convex_hull,
    "tsdf": estimate_tsdf,
    "voxel_carving": estimate_voxel_carving,
}


def run_eval(
    scan_dir: str | Path,
    methods: list[str],
    *,
    voxel_length: float = 0.002,
    sdf_trunc: float = 0.010,
    voxel_size: float = 0.0025,
    voxel_downsample: float = 0.0015,
    repair_mesh: bool = False,
) -> list[dict]:
    print(f"Loading scan: {scan_dir}", flush=True)
    scan = load_prepared_scan(scan_dir)
    print(f"  {len(scan.frames)} frames, gt={scan.gt_volume.get('volume_cm3')} cm³", flush=True)

    results = []
    for name in methods:
        if name not in METHODS:
            raise ValueError(f"Unknown method: {name}. Choose from {list(METHODS)}")
        print(f"Running {name}...", flush=True)
        kwargs = {}
        if name == "tsdf":
            kwargs = dict(voxel_length=voxel_length, sdf_trunc=sdf_trunc, repair_mesh=repair_mesh)
        elif name == "voxel_carving":
            kwargs = dict(voxel_size=voxel_size)
        elif name == "convex_hull":
            kwargs = dict(voxel_downsample=voxel_downsample)
        try:
            report = METHODS[name](scan_dir, **kwargs)
            status = "ok" if report.get("volume_m3") is not None else report.get("status", "failed")
            results.append({**report, "status": status})
            print(f"  {name} done ({status})", flush=True)
        except Exception as exc:
            print(f"  {name} FAILED: {exc}", flush=True)
            results.append(
                {
                    "method": name,
                    "scan_dir": str(scan_dir),
                    "status": "failed",
                    "error": str(exc),
                    "volume_m3": None,
                    "volume_cm3": None,
                    "relative_error_percent": None,
                }
            )

    out_dir = Path(scan_dir) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "predicted_cm3", "gt_cm3", "relative_error_percent", "status",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "method": r["method"],
                    "predicted_cm3": r.get("volume_cm3"),
                    "gt_cm3": r.get("gt_volume_cm3") or scan.gt_volume.get("volume_cm3"),
                    "relative_error_percent": r.get("relative_error_percent"),
                    "status": r.get("status"),
                }
            )

    print(f"{'method':<16} {'pred_cm3':>12} {'gt_cm3':>12} {'rel_err_%':>10} {'status':>8}")
    print("-" * 64)
    gt_cm3 = scan.gt_volume.get("volume_cm3")
    for r in results:
        pred = r.get("volume_cm3")
        rel = r.get("relative_error_percent")
        print(
            f"{r['method']:<16} "
            f"{pred if pred is not None else 'N/A':>12} "
            f"{gt_cm3 if gt_cm3 is not None else 'N/A':>12} "
            f"{rel if rel is not None else 'N/A':>10} "
            f"{r.get('status', ''):>8}"
        )
    print(f"\nSummary: {summary_path}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate volume methods on a prepared scan")
    parser.add_argument("--scan_dir", required=True)
    parser.add_argument("--methods", nargs="+", default=["convex_hull", "tsdf", "voxel_carving"])
    parser.add_argument("--voxel_length", type=float, default=0.002)
    parser.add_argument("--sdf_trunc", type=float, default=0.010)
    parser.add_argument("--voxel_size", type=float, default=0.0025)
    parser.add_argument(
        "--voxel_downsample",
        type=float,
        default=0.0015,
        help="Convex hull point-cloud downsample voxel size (m); increase if OOM/Killed",
    )
    parser.add_argument("--repair_mesh", action="store_true")
    args = parser.parse_args(argv)
    run_eval(
        args.scan_dir,
        args.methods,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        voxel_size=args.voxel_size,
        voxel_downsample=args.voxel_downsample,
        repair_mesh=args.repair_mesh,
    )


if __name__ == "__main__":
    main()

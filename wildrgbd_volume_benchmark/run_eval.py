"""Evaluate volume methods on a prepared WildRGB-D scene."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from wildrgbd_volume_benchmark.methods.convex_hull import estimate_convex_hull_volume
from wildrgbd_volume_benchmark.methods.tsdf import estimate_tsdf_volume
from wildrgbd_volume_benchmark.methods.voxel_carving import estimate_voxel_carving_volume
from wildrgbd_volume_benchmark.scan_io import load_prepared_scene

METHODS = {
    "convex_hull": estimate_convex_hull_volume,
    "tsdf": estimate_tsdf_volume,
    "voxel_carving": estimate_voxel_carving_volume,
}


def run_eval(
    prepared_scene_dir: str | Path,
    methods: list[str],
    *,
    voxel_length: float = 0.003,
    sdf_trunc: float = 0.015,
    voxel_size: float = 0.004,
    repair_mesh: bool = False,
) -> list[dict]:
    scene = load_prepared_scene(prepared_scene_dir)
    results = []
    for name in methods:
        if name not in METHODS:
            raise ValueError(f"Unknown method: {name}")
        kwargs: dict = {}
        if name == "tsdf":
            kwargs = dict(voxel_length=voxel_length, sdf_trunc=sdf_trunc, repair_mesh=repair_mesh)
        elif name == "voxel_carving":
            kwargs = dict(voxel_size=voxel_size)
        report = METHODS[name](prepared_scene_dir, **kwargs)
        results.append(report)

    out_dir = Path(prepared_scene_dir) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "pred_cm3", "pseudo_gt_cm3", "relative_error_percent", "status", "notes",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "method": r["method"],
                    "pred_cm3": r.get("volume_cm3"),
                    "pseudo_gt_cm3": r.get("pseudo_gt_volume_cm3") or scene.pseudo_gt.get("volume_cm3"),
                    "relative_error_percent": r.get("relative_error_percent"),
                    "status": r.get("status"),
                    "notes": r.get("notes", ""),
                }
            )

    print(f"{'method':<16} {'pred_cm3':>12} {'pseudo_gt':>12} {'rel_err_%':>10} {'status':>12}")
    print("-" * 70)
    pg = scene.pseudo_gt.get("volume_cm3")
    for r in results:
        print(
            f"{r['method']:<16} "
            f"{r.get('volume_cm3') or 'N/A':>12} "
            f"{pg or 'N/A':>12} "
            f"{r.get('relative_error_percent') or 'N/A':>10} "
            f"{r.get('status', ''):>12}"
        )
    print(f"\nSummary: {summary_path}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate WildRGB-D volume methods")
    parser.add_argument("--prepared_scene_dir", required=True)
    parser.add_argument("--methods", nargs="+", default=["convex_hull", "tsdf", "voxel_carving"])
    parser.add_argument("--voxel_length", type=float, default=0.003)
    parser.add_argument("--sdf_trunc", type=float, default=0.015)
    parser.add_argument("--voxel_size", type=float, default=0.004)
    parser.add_argument("--repair_mesh", action="store_true")
    args = parser.parse_args(argv)
    run_eval(
        args.prepared_scene_dir,
        args.methods,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        voxel_size=args.voxel_size,
        repair_mesh=args.repair_mesh,
    )


if __name__ == "__main__":
    main()

"""Compare predicted volumes/meshes against T-LESS GT for one prepared scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

METHOD_PRED_ASSETS = {
    "convex_hull": ("hull_mesh.ply", "mesh"),
    "tsdf": ("tsdf_mesh_cleaned.ply", "mesh"),
    "voxel_carving": ("carved_voxels.ply", "points"),
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _color_mesh(mesh: trimesh.Trimesh, rgba: tuple[int, int, int, int]) -> trimesh.Trimesh:
    m = mesh.copy()
    m.visual.vertex_colors = rgba
    return m


def _color_points(cloud: trimesh.PointCloud, rgba: tuple[int, int, int, int]) -> trimesh.PointCloud:
    c = cloud.copy()
    c.colors = [rgba[:3]] * len(c.vertices)
    return c


def _side_by_side(
    gt: trimesh.Trimesh,
    pred: trimesh.Trimesh | trimesh.PointCloud,
    gap_m: float = 0.02,
) -> trimesh.Scene:
    """Place GT on the left, prediction on the right (object frame, meters)."""
    gt_vis = _color_mesh(gt, (180, 180, 180, 255))
    extent = gt.bounds[1] - gt.bounds[0]
    shift = float(extent.max()) + gap_m

    if isinstance(pred, trimesh.PointCloud):
        pred_vis = _color_points(pred, (255, 120, 40, 255))
        pred_vis.vertices = pred_vis.vertices + np.array([shift, 0.0, 0.0])
        scene = trimesh.Scene()
        scene.add_geometry(gt_vis, node_name="gt_mesh")
        scene.add_geometry(pred_vis, node_name="prediction")
        return scene

    pred_vis = _color_mesh(pred, (255, 120, 40, 255))
    pred_vis.apply_translation([shift, 0.0, 0.0])
    scene = trimesh.Scene()
    scene.add_geometry(gt_vis, node_name="gt_mesh")
    scene.add_geometry(pred_vis, node_name="prediction")
    return scene


def _relative_error_percent(pred_m3: float | None, gt_m3: float | None) -> float | None:
    if pred_m3 is None or gt_m3 is None or gt_m3 <= 0:
        return None
    return 100.0 * abs(pred_m3 - gt_m3) / gt_m3


def compare_scan(
    scan_dir: str | Path,
    methods: list[str] | None = None,
    out_dir: str | Path | None = None,
    gap_m: float = 0.02,
    show: bool = False,
) -> Path:
    """
    Print GT vs predicted volumes and write side-by-side comparison assets.

    Returns path to comparison output directory.
    """
    root = Path(scan_dir).expanduser().resolve()
    gt_mesh_path = root / "gt_mesh.ply"
    gt_vol_path = root / "gt_volume.json"
    if not gt_mesh_path.is_file():
        raise FileNotFoundError(f"Missing GT mesh: {gt_mesh_path}")
    if not gt_vol_path.is_file():
        raise FileNotFoundError(f"Missing GT volume: {gt_vol_path}")

    gt_mesh = trimesh.load(gt_mesh_path, force="mesh", process=False)
    if not isinstance(gt_mesh, trimesh.Trimesh):
        raise ValueError(f"Expected mesh at {gt_mesh_path}")
    gt_vol = _load_json(gt_vol_path)

    cmp_dir = Path(out_dir).expanduser().resolve() if out_dir else root / "outputs" / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    available = []
    outputs_root = root / "outputs"
    for name in (methods or list(METHOD_PRED_ASSETS)):
        if name not in METHOD_PRED_ASSETS:
            continue
        asset, _ = METHOD_PRED_ASSETS[name]
        if (outputs_root / name / asset).is_file():
            available.append(name)

    rows: list[dict[str, Any]] = []
    print(f"\nScan: {root}")
    print(f"GT mesh: {gt_mesh_path}")
    print(f"GT type: {gt_vol.get('gt_type', '?')}  watertight={gt_vol.get('watertight')}  exact_gt={gt_vol.get('exact_gt')}")
    gt_m3 = gt_vol.get("volume_m3")
    gt_cm3 = gt_vol.get("volume_cm3")
    if gt_cm3 is None and gt_m3 is not None:
        gt_cm3 = gt_m3 * 1e6
    print(f"GT volume: {gt_cm3:.4f} cm³  ({gt_m3} m³)" if gt_cm3 is not None else "GT volume: unavailable")
    print()
    print(f"{'method':<16} {'pred_cm3':>12} {'gt_cm3':>12} {'abs_err_cm3':>12} {'rel_err_%':>10} {'status':>10}")
    print("-" * 78)

    for method in available:
        method_dir = outputs_root / method
        report_path = method_dir / "report.json"
        report = _load_json(report_path) if report_path.is_file() else {}

        pred_m3 = report.get("volume_m3")
        pred_cm3 = report.get("volume_cm3")
        rel = report.get("relative_error_percent")
        if rel is None:
            rel = _relative_error_percent(pred_m3, gt_m3)
        abs_cm3 = abs(pred_cm3 - gt_cm3) * 1e6 if pred_cm3 is not None and gt_m3 is not None else None
        status = report.get("status", "ok" if pred_m3 is not None else "unknown")

        print(
            f"{method:<16} "
            f"{pred_cm3 if pred_cm3 is not None else 'N/A':>12} "
            f"{gt_cm3 if gt_cm3 is not None else 'N/A':>12} "
            f"{abs_cm3 if abs_cm3 is not None else 'N/A':>12} "
            f"{rel if rel is not None else 'N/A':>10} "
            f"{status:>10}"
        )

        asset_name, asset_kind = METHOD_PRED_ASSETS[method]
        pred_path = method_dir / asset_name
        if asset_kind == "mesh":
            pred_geom = trimesh.load(pred_path, force="mesh", process=False)
            if not isinstance(pred_geom, trimesh.Trimesh):
                continue
        else:
            pred_geom = trimesh.load(pred_path, process=False)
            if not isinstance(pred_geom, trimesh.PointCloud):
                continue

        scene = _side_by_side(gt_mesh, pred_geom, gap_m=gap_m)
        side_path = cmp_dir / f"{method}_gt_vs_pred_side_by_side.ply"
        scene.export(side_path)

        rows.append(
            {
                "method": method,
                "gt_volume_m3": gt_m3,
                "gt_volume_cm3": gt_cm3,
                "pred_volume_m3": pred_m3,
                "pred_volume_cm3": pred_cm3,
                "abs_error_cm3": abs_cm3,
                "relative_error_percent": rel,
                "status": status,
                "gt_mesh": str(gt_mesh_path),
                "prediction_asset": str(pred_path),
                "side_by_side_ply": str(side_path),
                "report_json": str(report_path) if report_path.is_file() else None,
            }
        )

        if show:
            try:
                scene.show()
            except Exception as exc:
                print(f"  (Could not open viewer for {method}: {exc})")

    summary = {
        "scan_dir": str(root),
        "gt_volume": gt_vol,
        "error_definition": {
            "relative_error_percent": "100 * |pred_volume_m3 - gt_volume_m3| / gt_volume_m3",
            "abs_error_cm3": "|pred_volume_m3 - gt_volume_m3| * 1e6",
            "reference": "tless_volume_benchmark/scan_io.py gt_comparison_fields()",
        },
        "methods": rows,
        "view_instructions": {
            "side_by_side_ply": f"{cmp_dir}/<method>_gt_vs_pred_side_by_side.ply — gray=GT (left), orange=prediction (right)",
            "meshlab": "File → Import Mesh → open the side_by_side PLY",
            "blender": "File → Import → Stanford (.ply)",
            "open3d": "python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_triangle_mesh('PATH')])\"",
        },
    }
    summary_path = cmp_dir / "comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Error metrics (also in each outputs/<method>/report.json and outputs/summary.csv):")
    print("  rel_err_%  = 100 × |pred − GT| / GT   (volume in m³ internally)")
    print("  abs_err_cm3 = |pred − GT| × 10⁶")
    print()
    print("Visual comparison files (GT left / prediction right):")
    for row in rows:
        print(f"  {row['method']}: {row['side_by_side_ply']}")
    print(f"\nFull comparison index: {summary_path}")
    return cmp_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Print GT vs predicted volumes and export side-by-side mesh comparisons."
    )
    parser.add_argument("--scan_dir", required=True, help="Prepared scan (contains gt_mesh.ply)")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["convex_hull", "tsdf", "voxel_carving"],
        choices=list(METHOD_PRED_ASSETS),
    )
    parser.add_argument("--out_dir", default=None, help="Default: <scan_dir>/outputs/compare")
    parser.add_argument("--gap_m", type=float, default=0.02, help="Gap between GT and pred in side-by-side PLY (m)")
    parser.add_argument("--show", action="store_true", help="Try trimesh interactive viewer (needs display)")
    args = parser.parse_args(argv)
    compare_scan(args.scan_dir, args.methods, args.out_dir, gap_m=args.gap_m, show=args.show)


if __name__ == "__main__":
    main()

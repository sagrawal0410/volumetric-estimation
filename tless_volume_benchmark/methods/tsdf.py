"""TSDF volume estimation — NumPy backend by default, Open3D optional."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from tless_volume_benchmark.mesh_volume import clean_mesh
from tless_volume_benchmark.scan_io import (
    gt_comparison_fields,
    load_prepared_scan,
    resolve_output_dir,
    write_report,
)


def _largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    return max(parts, key=lambda m: len(m.faces)) if parts else mesh


def _estimate_tsdf_open3d(
    scan,
    out: Path,
    voxel_length: float,
    sdf_trunc: float,
    depth_trunc: float,
    repair_mesh: bool,
    verbose: bool,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict[str, Any], float | None]:
    from tless_volume_benchmark.methods.tsdf_open3d import run_open3d_tsdf

    return run_open3d_tsdf(
        scan, voxel_length, sdf_trunc, depth_trunc, repair_mesh=repair_mesh, verbose=verbose
    )


def estimate_tsdf(
    scan_dir: str | Path,
    voxel_length: float = 0.002,
    sdf_trunc: float = 0.010,
    depth_trunc: float = 5.0,
    output_dir: str | Path | None = None,
    repair_mesh: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    scan = load_prepared_scan(scan_dir)
    out = resolve_output_dir(scan.scan_dir, "tsdf", Path(output_dir) if output_dir else None)
    out.mkdir(parents=True, exist_ok=True)

    backend = os.environ.get("TLESS_TSDF_BACKEND", "numpy").lower()
    if verbose:
        print(f"  TSDF backend: {backend}", flush=True)

    if backend == "open3d":
        raw_mesh, cleaned, vol_meta, volume_m3 = _estimate_tsdf_open3d(
            scan, out, voxel_length, sdf_trunc, depth_trunc, repair_mesh, verbose
        )
    else:
        from tless_volume_benchmark.methods.tsdf_numpy import run_numpy_tsdf

        if verbose:
            print("  fusing depth frames (numpy)...", flush=True)
        raw_mesh, cleaned, vol_meta, volume_m3 = run_numpy_tsdf(
            scan, voxel_length, sdf_trunc, depth_trunc
        )
        if not vol_meta.get("watertight") and repair_mesh:
            try:
                trimesh.repair.fill_holes(cleaned)
                cleaned.fix_normals()
                cleaned.merge_vertices()
                if cleaned.is_watertight and cleaned.volume > 0:
                    volume_m3 = abs(float(cleaned.volume))
                    vol_meta["volume_source"] = "mesh_repaired"
                    vol_meta["watertight"] = True
            except Exception:
                pass

    raw_mesh.export(out / "tsdf_mesh_raw.ply")
    cleaned.export(out / "tsdf_mesh_cleaned.ply")

    watertight = bool(vol_meta.get("watertight", cleaned.is_watertight))
    report: dict[str, Any] = {
        "method": "tsdf",
        "scan_dir": str(scan.scan_dir),
        "volume_m3": volume_m3,
        "volume_cm3": volume_m3 * 1e6 if volume_m3 is not None else None,
        "watertight": watertight,
        "repaired_used": repair_mesh and watertight,
        "voxel_length": voxel_length,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
        "status": "ok" if volume_m3 is not None else "invalid_mesh",
        "outputs": {
            "tsdf_mesh_raw": str(out / "tsdf_mesh_raw.ply"),
            "tsdf_mesh_cleaned": str(out / "tsdf_mesh_cleaned.ply"),
        },
        **vol_meta,
    }
    if volume_m3 is None:
        report["warning"] = "TSDF volume unavailable"
    report.update(gt_comparison_fields(scan, volume_m3))
    write_report(out / "report.json", report)
    return report

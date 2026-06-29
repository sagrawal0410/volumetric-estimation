"""Synthetic volume smoke test."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from rich.console import Console

from rtabmap_volume.eval.synthetic_tests import (
    analytical_volume,
    make_cube,
    make_cylinder,
    make_partial_cube,
    make_sphere,
    make_u_shape,
    mesh_to_noisy_cloud,
)
from rtabmap_volume.pipeline import run_pipeline
from rtabmap_volume.volume.mesh_volume import compute_mesh_volume
from rtabmap_volume.volume.voxel_volume import compute_voxel_volumes, VoxelConfig
from rtabmap_volume.volume.convex_hull_volume import compute_convex_hull_volume
from rtabmap_volume.config import VoxelConfig

console = Console()
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "high_accuracy.yaml"
TOL = 0.08  # 8% tolerance for smoke test


def _check(name: str, pred: float | None, truth: float, tol: float = TOL) -> bool:
    if pred is None:
        console.print(f"[red]FAIL[/red] {name}: no prediction")
        return False
    err = abs(pred - truth) / truth
    ok = err <= tol
    color = "green" if ok else "red"
    console.print(f"[{color}]{'PASS' if ok else 'FAIL'}[/{color}] {name}: pred={pred:.4f} truth={truth:.4f} err={err*100:.1f}%")
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Synthetic volume smoke test")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    passed = 0
    total = 0

    # Watertight cube direct mesh
    cube = make_cube(1.0)
    est = compute_mesh_volume(cube)
    total += 1
    if _check("cube direct mesh", est.value_m3, 1.0, 0.01):
        passed += 1
    total += 1
    if est.reliable:
        console.print("[green]PASS[/green] cube marked reliable")
        passed += 1
    else:
        console.print("[red]FAIL[/red] cube should be reliable")

    # Sphere
    sphere = make_sphere(0.5)
    est_s = compute_mesh_volume(sphere)
    truth_s = analytical_volume("sphere", radius=0.5)
    total += 1
    if _check("sphere direct mesh", est_s.value_m3, truth_s, 0.05):
        passed += 1

    # Cylinder
    cyl = make_cylinder(0.3, 1.0)
    est_c = compute_mesh_volume(cyl)
    truth_c = analytical_volume("cylinder", radius=0.3, height=1.0)
    total += 1
    if _check("cylinder direct mesh", est_c.value_m3, truth_c, 0.05):
        passed += 1

    # Convex hull overestimates U-shape
    u = make_u_shape()
    u_vol = compute_mesh_volume(u)
    hull = compute_convex_hull_volume(mesh=u)
    total += 1
    if hull.value_m3 and u_vol.value_m3 and hull.value_m3 > u_vol.value_m3:
        console.print("[green]PASS[/green] convex hull > U-shape volume")
        passed += 1
    else:
        console.print("[red]FAIL[/red] convex hull should overestimate concave U-shape")

    # Voxel convergence on cube
    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(cube.vertices)
    pcd.points = o3d.utility.Vector3dVector(pts)
    vox_est, _ = compute_voxel_volumes(pcd=pcd, mesh=cube, cfg=VoxelConfig(voxel_sizes_m=[0.05, 0.02, 0.01]))
    total += 1
    if _check("voxel cube", vox_est.value_m3, 1.0, 0.15):
        passed += 1

    # Partial cube — confidence should drop
    partial = make_partial_cube()
    with tempfile.TemporaryDirectory() as td:
        ply = Path(td) / "partial.ply"
        partial.export(str(ply))
        out = Path(args.out) if args.out else Path(td) / "run"
        result = run_pipeline(ply, out, CONFIG, segmentation="none", overwrite=True, command="smoke partial")
        total += 1
        if result.get("confidence") in ("low", "medium"):
            console.print("[green]PASS[/green] partial scan low/medium confidence")
            passed += 1
        else:
            console.print("[red]FAIL[/red] partial scan should have reduced confidence")

    # Noisy point cloud pipeline
    noisy_pts = mesh_to_noisy_cloud(make_cube(0.5), n=10000, noise_std=0.002)
    pcd_n = o3d.geometry.PointCloud()
    pcd_n.points = o3d.utility.Vector3dVector(noisy_pts)
    default_cfg = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with tempfile.TemporaryDirectory() as td:
        ply = Path(td) / "noisy.ply"
        o3d.io.write_point_cloud(str(ply), pcd_n)
        out = Path(td) / "noisy_run"
        result = run_pipeline(ply, out, default_cfg, segmentation="none", overwrite=True, command="smoke noisy")
        total += 1
        pred = result.get("final_volume_m3")
        if pred and _check("noisy cube pipeline", pred, 0.125, 0.35):
            passed += 1

    console.print(f"\n[bold]Results: {passed}/{total} passed[/bold]")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

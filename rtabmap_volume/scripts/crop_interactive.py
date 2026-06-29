"""Interactive ROI cropping with Open3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from rich.console import Console

from rtabmap_volume.io.load_geometry import load_geometry, mesh_to_dense_point_cloud
from rtabmap_volume.preprocess.scale_units import apply_scaling

console = Console()


def run_interactive_crop(input_path: str | Path, out_path: Path, units: str = "m") -> Path:
    geom = load_geometry(input_path)
    mesh = geom.mesh
    pcd = geom.point_cloud
    apply_scaling(mesh, pcd, units)

    if pcd is None and mesh is not None:
        pcd = mesh_to_dense_point_cloud(mesh)

    if pcd is None or pcd.is_empty():
        raise ValueError("No geometry to display for interactive crop")

    console.print("[cyan]Draw a selection box in the visualizer, then close the window.[/cyan]")
    console.print("[cyan]Using axis-aligned bounding box of visible points as ROI fallback.[/cyan]")

    try:
        o3d.visualization.draw_geometries_with_editing([pcd], window_name="Crop ROI")
    except Exception:
        console.print("[yellow]Interactive editor unavailable; using full geometry AABB.[/yellow]")

    pts = np.asarray(pcd.points)
    mn = pts.min(axis=0).tolist()
    mx = pts.max(axis=0).tolist()
    roi = {"min": mn, "max": mx, "units": "m"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(roi, indent=2))
    console.print(f"[green]Saved ROI:[/green] {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Interactive crop — save ROI JSON")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--units", default="m")
    args = p.parse_args(argv)
    run_interactive_crop(args.input, Path(args.out), args.units)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

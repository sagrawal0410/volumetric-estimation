"""View a PLY mesh (GT, TSDF output, side-by-side comparison, etc.)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


def show_mesh(path: str | Path, backend: str = "auto") -> None:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Mesh not found: {path}")

    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        mesh = trimesh.util.concatenate(geoms) if geoms else None
    else:
        mesh = loaded

    if mesh is None or (not hasattr(mesh, "vertices") or len(mesh.vertices) == 0):
        raise ValueError(f"No geometry to display in {path}")

    print(f"Loaded: {path}")
    print(f"  vertices: {len(mesh.vertices):,}  faces: {getattr(mesh, 'faces', np.array([])).shape[0]:,}")
    if hasattr(mesh, "is_watertight") and mesh.is_watertight:
        print(f"  volume: {abs(float(mesh.volume)) * 1e6:.4f} cm³")

    if backend == "open3d" or (backend == "auto" and sys.platform.startswith("linux")):
        try:
            import open3d as o3d

            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d_mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries(
                [o3d_mesh],
                window_name=str(path.name),
                width=1280,
                height=720,
            )
            return
        except Exception as exc:
            print(f"Open3D viewer failed ({exc}), trying trimesh...", file=sys.stderr)

    # trimesh / pyglet window (works on desktop; needs DISPLAY on Linux)
    mesh.show(smooth=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive viewer for PLY meshes")
    parser.add_argument("mesh_path", type=Path, help="Path to .ply file")
    parser.add_argument(
        "--backend",
        choices=["auto", "trimesh", "open3d"],
        default="auto",
        help="auto prefers open3d on Linux, trimesh elsewhere",
    )
    args = parser.parse_args(argv)
    show_mesh(args.mesh_path, backend=args.backend)


if __name__ == "__main__":
    main()

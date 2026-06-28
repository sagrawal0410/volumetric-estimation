"""Open3D scene visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

from volrecon.visualization.mesh_viewer import show_meshes


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize reconstruction outputs.")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--gt_mesh", type=Path, default=None)
    parser.add_argument("--pointcloud", type=Path, default=None)
    args = parser.parse_args()
    show_meshes(args.mesh, args.gt_mesh, args.pointcloud)


if __name__ == "__main__":
    main()

"""Point cloud viewer helper."""

from __future__ import annotations

from pathlib import Path

import open3d as o3d


def show_pointclouds(*paths: Path) -> None:
    geoms = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
    for i, p in enumerate(paths):
        if not Path(p).exists():
            continue
        pc = o3d.io.read_point_cloud(str(p))
        c = colors[i % len(colors)]
        pc.paint_uniform_color(c)
        geoms.append(pc)
    if not geoms:
        raise FileNotFoundError("No point clouds found")
    o3d.visualization.draw_geometries(geoms)

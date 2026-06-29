"""Open3D visualization helpers."""

from __future__ import annotations

import open3d as o3d


def try_offscreen() -> bool:
    try:
        o3d.visualization.rendering.OffscreenRenderer(64, 64)
        return True
    except Exception:
        return False

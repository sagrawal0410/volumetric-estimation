"""Check native dependencies for T-LESS volume benchmark."""

from __future__ import annotations

import platform
import subprocess
import sys


_NUMPY_TSDF_PROBE = """
import numpy as np
import trimesh
from tless_volume_benchmark.methods.tsdf_numpy import fuse_tsdf_grid, tsdf_grid_to_mesh
from tless_volume_benchmark.scan_io import PreparedScan, PreparedFrame

K = np.array([[200,0,64],[0,200,64],[0,0,1]], float)
T = np.eye(4)
depth = np.zeros((128,128), np.float32)
depth[40:88,40:88] = 0.45
mask = depth > 0
frame = PreparedFrame(0, np.zeros((128,128,3), np.uint8), depth, mask, K, T, {})
scan = PreparedScan(__import__('pathlib').Path('.'), [frame], {'volume_m3': 0.001}, None)
tsdf, w, origin, meta = fuse_tsdf_grid(scan, voxel_length=0.004, sdf_trunc=0.02)
mesh = tsdf_grid_to_mesh(tsdf, origin, meta['voxel_length_m'])
print(f'ok faces={len(mesh.faces)} backend=numpy')
"""


def _probe(label: str, code: str) -> tuple[str, str]:
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
            env={**__import__("os").environ, "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parents[1])},
        )
        if r.returncode == 0:
            return "ok", (r.stdout or "").strip() or label
        if r.returncode < 0:
            sig = -r.returncode
            return "fail", f"crashed (signal {sig})"
        msg = (r.stderr or r.stdout or "failed").strip().splitlines()
        return "fail", (msg[-1] if msg else "failed")[:240]
    except subprocess.TimeoutExpired:
        return "fail", "timed out"
    except Exception as exc:
        return "fail", str(exc)


def main() -> int:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    print(f"Python: {sys.executable}")
    print(f"Machine: {platform.machine()}  ({platform.platform()})")
    print(f"Repo root: {root}")
    print()

    checks = [
        ("numpy", "import numpy as np; print(np.__version__)"),
        ("opencv", "import cv2; print(cv2.__version__)"),
        ("trimesh", "import trimesh; print(trimesh.__version__)"),
        ("scipy", "import scipy; print(scipy.__version__)"),
        ("scikit-image", "import skimage; print(skimage.__version__)"),
        ("numpy_tsdf", _NUMPY_TSDF_PROBE),
        ("open3d_import", "import open3d as o3d; print(o3d.__version__)"),
    ]
    failed = 0
    for name, code in checks:
        status, detail = _probe(name, code)
        mark = "OK" if status == "ok" else "FAIL"
        if status != "ok":
            failed += 1
        print(f"[{mark}] {name}: {detail}")

    print()
    if failed:
        print("TSDF default backend is numpy (TLESS_TSDF_BACKEND=numpy). Open3D is optional.")
        print("  pip install scikit-image   # marching cubes for TSDF mesh export")
        print("  export TLESS_TSDF_BACKEND=open3d   # only if open3d integrate works on your machine")
        return 1
    print("All checks passed. TSDF uses numpy backend by default (no Open3D required).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

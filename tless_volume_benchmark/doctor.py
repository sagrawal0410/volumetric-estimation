"""Check native dependencies for T-LESS volume benchmark."""

from __future__ import annotations

import platform
import subprocess
import sys


_OPEN3D_TSDF_PROBE = """
import numpy as np
import open3d as o3d

h, w = 64, 64
depth = np.zeros((h, w), dtype=np.float32)
depth[20:44, 20:44] = 0.45
color = o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8))
dimg = o3d.geometry.Image(depth)
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, dimg, depth_scale=1.0, depth_trunc=2.0, convert_rgb_to_intensity=False
)
vol = o3d.pipelines.integration.UniformTSDFVolume(
    0.6,
    64,
    0.04,
    o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    origin=[-0.3, -0.3, 0.2],
)
intr = o3d.camera.PinholeCameraIntrinsic(w, h, 200.0, 200.0, w / 2, h / 2)
extr = np.eye(4, dtype=np.float64)
vol.integrate(rgbd, intr, extr)
mesh = vol.extract_triangle_mesh()
print(f"ok vertices={len(mesh.vertices)}")
"""


def _probe(label: str, code: str) -> tuple[str, str]:
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode == 0:
            return "ok", (r.stdout or "").strip() or label
        if r.returncode < 0:
            sig = -r.returncode
            return "fail", f"crashed (signal {sig}, often segfault in native code)"
        msg = (r.stderr or r.stdout or "failed").strip().splitlines()
        return "fail", (msg[-1] if msg else "failed")[:240]
    except subprocess.TimeoutExpired:
        return "fail", "timed out (possible hang/crash)"
    except Exception as exc:
        return "fail", str(exc)


def main() -> int:
    print(f"Python: {sys.executable}")
    print(f"Machine: {platform.machine()}  ({platform.platform()})")
    print()

    checks = [
        ("numpy", "import numpy as np; print(np.__version__)"),
        ("opencv", "import cv2; print(cv2.__version__)"),
        ("trimesh", "import trimesh; print(trimesh.__version__)"),
        (
            "scipy",
            "import scipy; from scipy.spatial import cKDTree; import numpy as np; "
            "cKDTree(np.zeros((4,3))); print(scipy.__version__)",
        ),
        ("open3d_import", "import open3d as o3d; print(o3d.__version__)"),
        ("open3d_tsdf_integrate", _OPEN3D_TSDF_PROBE),
        ("sklearn", "import sklearn; print(sklearn.__version__)"),
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
        print("Some checks failed.")
        print("Open3D import OK but TSDF integrate crash → broken/incompatible Open3D build.")
        print("Try:")
        print("  pip uninstall -y open3d && pip install 'open3d>=0.17,<0.20'")
        print("  export TLESS_TSDF_BACKEND=uniform   # default; safer than scalable")
        print("  --methods convex_hull voxel_carving   # skip TSDF")
        return 1
    print("All checks passed (including Open3D TSDF integrate).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

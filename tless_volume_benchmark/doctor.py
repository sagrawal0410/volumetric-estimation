"""Check native dependencies for T-LESS volume benchmark."""

from __future__ import annotations

import platform
import subprocess
import sys


def _probe(label: str, code: str) -> tuple[str, str]:
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode == 0:
            return "ok", (r.stdout or "").strip() or label
        msg = (r.stderr or r.stdout or "failed").strip().splitlines()[-1]
        return "fail", msg[:200]
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
        ("scipy", "import scipy; from scipy.spatial import cKDTree; import numpy as np; cKDTree(np.zeros((4,3))); print(scipy.__version__)"),
        ("open3d", "import open3d as o3d; print(o3d.__version__)"),
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
        print("Some checks failed. On Apple Silicon, recreate the venv with native arm64 Python:")
        print("  rm -rf .venv")
        print("  python3 -m venv .venv && source .venv/bin/activate")
        print("  pip install -U pip && pip install -r requirements.txt")
        print()
        print("Workarounds:")
        print("  --methods convex_hull voxel_carving     # skip Open3D TSDF")
        print("  export TLESS_SKIP_OPEN3D=1              # force-skip TSDF")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

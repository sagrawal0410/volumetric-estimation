"""Run directory management and environment metadata."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

from volrecon.io.json_io import write_json


def save_environment(run_dir: Path, extra: dict | None = None) -> None:
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        env["cuda_available"] = False

    try:
        from volrecon.camera.zed_backend import get_sl_module

        sl = get_sl_module()
        env["zed_sdk_version"] = sl.get_sdk_version() if hasattr(sl, "get_sdk_version") else "unknown"
    except Exception as exc:  # noqa: BLE001
        env["zed_sdk_version"] = f"unavailable: {exc}"

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        env["git_commit"] = commit
    except Exception:  # noqa: BLE001
        env["git_commit"] = None

    if extra:
        env.update(extra)
    write_json(run_dir / "environment.json", env)


def run_paths(scene_dir: Path, method: str = "plain_tsdf") -> dict[str, Path]:
    base = scene_dir / "runs" / method
    return {
        "depth_predictions": base / "depth_predictions",
        "reconstructions": base / "reconstructions",
        "eval_report": base / "eval_report",
        "uncertainty": scene_dir / "runs" / "weighted_tsdf" / "uncertainty",
        "comparison": scene_dir / "runs" / "weighted_tsdf" / "comparison_report",
    }

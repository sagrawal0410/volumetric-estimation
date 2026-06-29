"""Save pipeline outputs to structured run directories."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import pandas as pd
import trimesh
import yaml


def ensure_run_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": out_dir,
        "inputs": out_dir / "inputs",
        "processed": out_dir / "processed",
        "reports": out_dir / "reports",
        "screenshots": out_dir / "screenshots",
        "logs": out_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def save_yaml(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def save_warnings(warnings: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for w in warnings:
            f.write(w + "\n")


def save_mesh(mesh: trimesh.Trimesh | None, path: Path) -> None:
    if mesh is None or len(mesh.vertices) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def save_mesh_o3d(mesh: o3d.geometry.TriangleMesh | None, path: Path) -> None:
    if mesh is None or mesh.is_empty():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh)


def save_point_cloud(pcd: o3d.geometry.PointCloud | None, path: Path) -> None:
    if pcd is None or pcd.is_empty():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def copy_input(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def save_voxel_grid(occupied: np.ndarray, voxel_size: float, origin: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, occupied=occupied, voxel_size=voxel_size, origin=origin)


def save_volume_csv(volume_result: dict[str, Any], path: Path) -> None:
    rows = []
    all_est = volume_result.get("all_estimates", {})
    for name, est in all_est.items():
        if isinstance(est, dict):
            rows.append(
                {
                    "estimator": name,
                    "value_m3": est.get("value_m3"),
                    "value_liters": est.get("value_liters"),
                    "reliable": est.get("reliable"),
                }
            )
    rows.append(
        {
            "estimator": "final_consensus",
            "value_m3": volume_result.get("final_volume_m3"),
            "value_liters": volume_result.get("final_volume_liters"),
            "reliable": volume_result.get("confidence") == "high",
        }
    )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def init_processing_log(command: str) -> dict[str, Any]:
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "steps": [],
        "finished_at": None,
    }


def append_log_step(log: dict[str, Any], step: str, details: dict[str, Any] | None = None) -> None:
    entry: dict[str, Any] = {"step": step, "timestamp": datetime.now(timezone.utc).isoformat()}
    if details:
        entry["details"] = details
    log["steps"].append(entry)

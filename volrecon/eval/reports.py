"""Evaluation report aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from volrecon.io.json_io import write_json


def write_scene_eval_bundle(out_dir: Path, metrics: dict[str, Any], volume: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "volume.json", volume)


def summarize_run(scene_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not scene_reports:
        return {"num_scenes": 0}
    chamfers = [r["metrics"]["chamfer_l1_m"] for r in scene_reports if "metrics" in r]
    vol_errs = [r["volume"]["rel_volume_error_percent"] for r in scene_reports if "volume" in r]
    return {
        "num_scenes": len(scene_reports),
        "mean_chamfer_l1_m": float(sum(chamfers) / len(chamfers)) if chamfers else None,
        "mean_rel_volume_error_percent": float(sum(vol_errs) / len(vol_errs)) if vol_errs else None,
        "scenes": scene_reports,
    }

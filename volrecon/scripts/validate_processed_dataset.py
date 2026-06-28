"""Validate processed dataset manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from volrecon.config import PROJECT_ROOT
from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.datasets.preprocessing import foundation_stereo_usable
from volrecon.io.image_io import image_shape, read_image
from volrecon.io.json_io import read_jsonl

console = Console()


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def validate_manifest(manifest_path: Path) -> dict:
    records = [ViewRecord.from_dict(r) for r in read_jsonl(manifest_path)]
    warnings: list[str] = []
    errors: list[str] = []

    scenes = {r.scene_id for r in records}
    n_true_stereo = 0
    n_synthetic = 0
    n_gt_depth = 0
    n_world_pose = 0
    n_fs_usable = 0

    for rec in records:
        if "gt_depth" in rec.inference_allowed_modalities:
            errors.append(f"{rec.scene_id}/{rec.view_id}: gt_depth in inference_allowed_modalities")

        if rec.gt_depth_path:
            n_gt_depth += 1
            p = _resolve(str(rec.gt_depth_path))
            if p is None or not p.exists():
                errors.append(f"Missing gt_depth: {rec.gt_depth_path}")

        for label, p_raw in [
            ("rgb", rec.rgb_path),
            ("left", rec.left_path),
            ("right", rec.right_path),
        ]:
            if p_raw is None:
                continue
            p = _resolve(str(p_raw))
            if p is None or not p.exists():
                errors.append(f"Missing {label}: {p_raw}")
            elif rec.K is not None:
                w, h = image_shape(p)
                fx, fy, cx, cy = rec.K[0, 0], rec.K[1, 1], rec.K[0, 2], rec.K[1, 2]
                if abs(cx - w / 2) > w * 0.25 or abs(cy - h / 2) > h * 0.25:
                    warnings.append(f"{rec.scene_id}/{rec.view_id}: K principal point far from image center")

        if rec.stereo and rec.stereo.has_true_stereo:
            n_true_stereo += 1
            if rec.left_path and rec.right_path:
                lp = _resolve(str(rec.left_path))
                rp = _resolve(str(rec.right_path))
                if lp and rp and lp.exists() and rp.exists():
                    if image_shape(lp) != image_shape(rp):
                        errors.append(f"{rec.scene_id}/{rec.view_id}: left/right dimension mismatch")
            if rec.stereo.baseline_m is None and not rec.stereo.synthetic:
                warnings.append(f"{rec.scene_id}/{rec.view_id}: stereo pair without baseline metadata")

        if rec.synthetic or (rec.stereo and rec.stereo.synthetic):
            n_synthetic += 1

        if rec.T_world_cam is not None:
            n_world_pose += 1

        if foundation_stereo_usable(rec):
            n_fs_usable += 1

        for op in rec.object_poses:
            mp = _resolve(str(op.model_path))
            if mp is None or not mp.exists():
                errors.append(f"Missing object model obj_id={op.obj_id}: {op.model_path}")

        if rec.dataset == "bop_tless" and rec.gt_depth_path:
            meta_path = _resolve(str(rec.gt_depth_path))
            if meta_path:
                view_meta = meta_path.parent / "meta.json"
                if view_meta.exists():
                    import json

                    meta = json.loads(view_meta.read_text())
                    if meta.get("original_units") == "mm" and not rec.unit_conversion_applied:
                        warnings.append(f"{rec.scene_id}/{rec.view_id}: mm units not converted")

    return {
        "records": records,
        "scenes": scenes,
        "n_views": len(records),
        "n_true_stereo": n_true_stereo,
        "n_synthetic": n_synthetic,
        "n_gt_depth": n_gt_depth,
        "n_world_pose": n_world_pose,
        "n_fs_usable": n_fs_usable,
        "warnings": warnings,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed dataset manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    result = validate_manifest(args.manifest.resolve())

    table = Table(title="Processed Dataset Validation Summary")
    table.add_column("Metric")
    table.add_column("Count")
    table.add_row("Scenes", str(len(result["scenes"])))
    table.add_row("Views", str(result["n_views"]))
    table.add_row("True stereo", str(result["n_true_stereo"]))
    table.add_row("Synthetic stereo", str(result["n_synthetic"]))
    table.add_row("With GT depth (eval-only)", str(result["n_gt_depth"]))
    table.add_row("With world camera pose", str(result["n_world_pose"]))
    table.add_row("FoundationStereo usable", str(result["n_fs_usable"]))
    table.add_row("Errors", str(len(result["errors"])))
    table.add_row("Warnings", str(len(result["warnings"])))
    console.print(table)

    if result["errors"]:
        console.print("\n[red]Errors:[/red]")
        for e in result["errors"][:50]:
            console.print(f"  - {e}")
    if result["warnings"]:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in result["warnings"][:50]:
            console.print(f"  - {w}")

    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

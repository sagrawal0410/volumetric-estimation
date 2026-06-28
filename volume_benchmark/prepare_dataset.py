"""CLI for converting raw datasets into normalized prepared scans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from volume_benchmark.common.io import validate_prepared_scan
from volume_benchmark.common.mesh_volume import (
    compute_mesh_volume_m3,
    load_mesh_as_meters,
    write_gt_volume_json,
)
from volume_benchmark.datasets.bop_adapter import prepare_bop_scan
from volume_benchmark.datasets.ycb_adapter import prepare_ycb_scan


def _parse_frame_triplets(raw_frames: list[list[str]]) -> list[tuple[Path, Path, Path]]:
    triplets = []
    for idx, item in enumerate(raw_frames):
        if len(item) != 3:
            raise ValueError(
                f"Frame triplet {idx} must have 3 paths [depth, mask, pose/camera], got {len(item)}"
            )
        triplets.append(tuple(Path(p) for p in item))
    return triplets


def _load_manifest(manifest_path: Path) -> dict:
    suffix = manifest_path.suffix.lower()
    text = manifest_path.read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path}")
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a mapping")
    return data


def _prepare_from_manifest(manifest: dict) -> Path:
    dataset = manifest.get("dataset")
    if not dataset:
        raise ValueError("Manifest must include 'dataset' (bop or ycb)")

    output_dir = Path(manifest["output_dir"])
    repair_mesh = bool(manifest.get("repair_mesh", False))
    mesh_units = manifest.get("mesh_units", "auto")
    metadata = manifest.get("metadata", {})

    mesh_path = Path(manifest["mesh_path"])
    frames = _parse_frame_triplets(manifest["frames"])

    if dataset == "bop":
        return prepare_bop_scan(
            output_dir=output_dir,
            mesh_path=mesh_path,
            frames=frames,
            mesh_units=mesh_units,
            repair_mesh=repair_mesh,
            depth_scale=float(manifest.get("depth_scale", 0.001)),
            metadata=metadata,
        )
    if dataset == "ycb":
        import numpy as np

        K = np.array(manifest["K"], dtype=np.float64).reshape(3, 3)
        return prepare_ycb_scan(
            output_dir=output_dir,
            mesh_path=mesh_path,
            K=K,
            frames=frames,
            mesh_units=mesh_units,
            repair_mesh=repair_mesh,
            depth_scale=float(manifest.get("depth_scale", 0.001)),
            pose_is_cam_to_object=bool(manifest.get("pose_is_cam_to_object", True)),
            metadata=metadata,
        )

    raise ValueError(f"Unknown dataset: {dataset}")


def _prepare_mesh_only(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    mesh = load_mesh_as_meters(args.mesh_path, source_units=args.mesh_units)
    volume_m3, watertight, gt_type = compute_mesh_volume_m3(mesh, repair=args.repair_mesh)

    output_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(args.mesh_path, output_dir / "gt_mesh.ply")
    write_gt_volume_json(
        output_dir / "gt_volume.json",
        volume_m3=volume_m3,
        method=gt_type,
        watertight=watertight,
        source_mesh=args.mesh_path,
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare normalized volume-benchmark scan directories from raw datasets."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("from-manifest", help="Prepare scan from YAML/JSON manifest")
    manifest.add_argument("manifest", type=Path, help="Path to manifest (.yaml/.json)")
    manifest.add_argument(
        "--validate",
        action="store_true",
        help="Run validate_prepared_scan after writing",
    )

    mesh_only = sub.add_parser(
        "mesh-gt",
        help="Write gt_volume.json for a mesh without RGB-D frames",
    )
    mesh_only.add_argument("mesh_path", type=Path)
    mesh_only.add_argument("output_dir", type=Path)
    mesh_only.add_argument("--mesh-units", choices=["m", "mm", "auto"], default="auto")
    mesh_only.add_argument("--repair-mesh", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "from-manifest":
            out = _prepare_from_manifest(_load_manifest(args.manifest))
        elif args.command == "mesh-gt":
            out = _prepare_mesh_only(args)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2

        if getattr(args, "validate", False):
            errors = validate_prepared_scan(out)
            if errors:
                print(f"Validation failed for {out}:", file=sys.stderr)
                for err in errors:
                    print(f"  - {err}", file=sys.stderr)
                return 1

        print(str(out))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

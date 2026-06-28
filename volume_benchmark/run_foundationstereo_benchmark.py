"""End-to-end FoundationStereo volume benchmark (T-LESS, WildRGB-D, BOP, YCB)."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from volume_benchmark.common.io import load_prepared_scan
from volume_benchmark.run_eval import METHODS, _run_method
from volume_benchmark.stereo.foundation_stereo_backend import FoundationStereoBackend
from volume_benchmark.stereo.fs_prepare_scan import fs_stereo_to_rgbd_scan


def _prepare_stereo_scan(args: argparse.Namespace, stereo_dir: Path) -> Path:
    dataset = args.dataset
    if dataset in ("bop_stereo_rendered", "tless_stereo_rendered"):
        from volume_benchmark.datasets.tless_stereo_adapter import prepare_tless_stereo_rendered

        return prepare_tless_stereo_rendered(
            dataset_root=args.dataset_root,
            split=args.split,
            object_id=args.object_id,
            out_dir=stereo_dir,
            baseline_m=args.baseline_m,
            num_views=args.num_views,
            min_visib_fract=args.min_visib_fract,
        )
    if dataset == "ycb_stereo_rendered":
        from volume_benchmark.datasets.ycb_stereo_adapter import prepare_ycb_stereo_rendered

        return prepare_ycb_stereo_rendered(
            object_root=args.object_root,
            out_dir=stereo_dir,
            baseline_m=args.baseline_m,
            num_views=args.num_views,
        )
    if dataset == "wildrgbd_stereo_rendered":
        from volume_benchmark.datasets.wildrgbd_stereo_adapter import prepare_wildrgbd_stereo_rendered

        if args.prepared_scene_dir is None:
            raise ValueError(
                "wildrgbd_stereo_rendered requires --prepared_scene_dir "
                "(output of wildrgbd_volume_benchmark.prepare_scene)"
            )
        return prepare_wildrgbd_stereo_rendered(
            prepared_scene_dir=args.prepared_scene_dir,
            out_dir=stereo_dir,
            baseline_m=args.baseline_m,
        )
    if dataset == "bigbird_stereo_rendered":
        from volume_benchmark.datasets.bigbird_stereo_adapter import prepare_bigbird_stereo_rendered

        return prepare_bigbird_stereo_rendered(
            object_root=args.object_root,
            out_dir=stereo_dir,
            baseline_m=args.baseline_m,
            num_views=args.num_views,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def _load_stereo_metadata(stereo_dir: Path) -> dict[str, Any]:
    meta_path = stereo_dir / "metadata.json"
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _run_volume_methods(scan_dir: Path, methods: list[str], output_dir: Path, args: argparse.Namespace) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    eval_args = argparse.Namespace(
        output_dir=output_dir,
        voxel_downsample=args.voxel_downsample,
        voxel_length=args.voxel_length,
        voxel_size=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        depth_trunc=args.depth_trunc,
        depth_tolerance=args.depth_tolerance,
        padding=args.padding,
        min_views_checked=args.min_views_checked,
        repair_mesh=args.repair_mesh,
    )
    for method in methods:
        try:
            report = _run_method(method, scan_dir, eval_args)
            rows.append(report)
        except Exception as exc:
            rows.append({"method": method, "error": str(exc), "scan_dir": str(scan_dir)})
    return rows


def _dataset_depth_baseline_scan(args: argparse.Namespace, out_root: Path) -> Path | None:
    """Optional baseline using dataset-provided depth (not FoundationStereo)."""
    if not args.compare_dataset_depth:
        return None

    baseline_dir = out_root / "prepared_depth_dataset"
    if args.dataset_depth_scan_dir is not None:
        src = Path(args.dataset_depth_scan_dir).resolve()
        if baseline_dir.exists():
            shutil.rmtree(baseline_dir)
        shutil.copytree(src, baseline_dir)
        return baseline_dir

    if args.dataset in ("bop_stereo_rendered", "tless_stereo_rendered"):
        from tless_volume_benchmark.tless_prepare import prepare_tless_scan

        return prepare_tless_scan(
            dataset_root=args.dataset_root,
            split=args.split,
            object_id=args.object_id,
            min_visib_fract=args.min_visib_fract,
            num_views=args.num_views,
            out_dir=baseline_dir,
        )
    if args.dataset == "wildrgbd_stereo_rendered":
        if args.prepared_scene_dir is None:
            raise ValueError("--prepared_scene_dir required for WildRGB-D dataset-depth baseline")
        from wildrgbd_volume_benchmark.scan_io import export_sampled_as_volume_benchmark_scan

        return export_sampled_as_volume_benchmark_scan(args.prepared_scene_dir, baseline_dir)

    return None


def _flatten_results(
    depth_backend: str,
    stereo_meta: dict[str, Any],
    fs_meta: dict[str, Any],
    method_rows: list[dict],
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in method_rows:
        if "error" in row:
            flat.append(
                {
                    "depth_backend": depth_backend,
                    "method": row.get("method"),
                    "error": row["error"],
                    **(extra or {}),
                }
            )
            continue
        gt_cm3 = row.get("gt_volume_cm3")
        pred_cm3 = (row.get("volume_m3") or 0) * 1e6 if row.get("volume_m3") is not None else None
        flat.append(
            {
                "depth_backend": depth_backend,
                "method": row.get("method"),
                "gt_volume_cm3": gt_cm3,
                "pred_volume_cm3": pred_cm3,
                "relative_error_percent": row.get("relative_error_percent"),
                "source_mode": stereo_meta.get("source_mode"),
                "baseline_m": stereo_meta.get("baseline_m") or fs_meta.get("baseline_m"),
                "model_variant": fs_meta.get("model_variant"),
                "checkpoint_path": fs_meta.get("checkpoint_path"),
                "num_views": len(row.get("frames_used", [])) or stereo_meta.get("num_views"),
                **(extra or {}),
            }
        )
    return flat


def run_benchmark(args: argparse.Namespace) -> Path:
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    stereo_dir = out_root / "prepared_stereo"
    if stereo_dir.exists() and args.overwrite:
        shutil.rmtree(stereo_dir)
    if not stereo_dir.is_dir() or not list((stereo_dir / "frames").glob("frame_*_left.png")):
        _prepare_stereo_scan(args, stereo_dir)
    stereo_meta = _load_stereo_metadata(stereo_dir)

    fs_dir = out_root / "prepared_depth_fs"
    backend = FoundationStereoBackend(
        repo_path=args.foundationstereo_repo,
        checkpoint_path=args.checkpoint,
        variant=args.variant,
        device=args.device,
        max_input_size=tuple(args.max_input_size) if args.max_input_size else None,
    )
    fs_stereo_to_rgbd_scan(
        stereo_dir,
        fs_dir,
        backend,
        max_depth_m=args.max_depth_m,
        save_debug=not args.no_debug,
    )
    fs_scan = load_prepared_scan(fs_dir)
    fs_meta = fs_scan.metadata

    fs_outputs = out_root / "outputs" / "foundationstereo"
    fs_rows = _run_volume_methods(fs_dir, args.methods, fs_outputs, args)

    all_rows = _flatten_results("foundationstereo", stereo_meta, fs_meta, fs_rows)

    dataset_scan = _dataset_depth_baseline_scan(args, out_root)
    if dataset_scan is not None:
        ds_outputs = out_root / "outputs" / "dataset_depth"
        ds_rows = _run_volume_methods(dataset_scan, args.methods, ds_outputs, args)
        all_rows.extend(
            _flatten_results(
                "dataset_depth",
                {"source_mode": "dataset_provided_depth"},
                {},
                ds_rows,
            )
        )

    summary_path = out_root / "summary.csv"
    if all_rows:
        fieldnames: list[str] = []
        for row in all_rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    manifest = {
        "out_root": str(out_root),
        "dataset": args.dataset,
        "stereo_scan_dir": str(stereo_dir),
        "foundationstereo_scan_dir": str(fs_dir),
        "source_mode": stereo_meta.get("source_mode"),
        "baseline_m": stereo_meta.get("baseline_m"),
        "checkpoint": str(args.checkpoint),
        "model_variant": args.variant,
        "num_views": stereo_meta.get("num_views", len(fs_scan.frames)),
        "results": all_rows,
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FoundationStereo end-to-end volume benchmark")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "bop_stereo_rendered",
            "tless_stereo_rendered",
            "ycb_stereo_rendered",
            "wildrgbd_stereo_rendered",
            "bigbird_stereo_rendered",
        ],
    )
    parser.add_argument("--dataset_root", type=Path, default=None, help="BOP/T-LESS dataset root")
    parser.add_argument("--split", default="test_primesense")
    parser.add_argument("--object_id", type=int, default=None)
    parser.add_argument("--object_root", type=Path, default=None, help="YCB/BigBIRD object root")
    parser.add_argument("--prepared_scene_dir", type=Path, default=None, help="WildRGB-D prepared scene")
    parser.add_argument("--num_views", type=int, default=5)
    parser.add_argument("--baseline_m", type=float, default=0.12)
    parser.add_argument("--min_visib_fract", type=float, default=0.5)
    parser.add_argument("--foundationstereo_repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", default="fast", choices=["fast", "full"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_depth_m", type=float, default=5.0)
    parser.add_argument("--max_input_size", type=int, nargs=2, default=None)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["convex_hull", "tsdf", "voxel_carving"], choices=sorted(METHODS))
    parser.add_argument("--compare-dataset-depth", action="store_true", help="Also run volume methods on dataset depth")
    parser.add_argument("--dataset-depth-scan-dir", type=Path, default=None, help="Existing prepared RGB-D scan with dataset depth")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--voxel-downsample", type=float, default=0.002)
    parser.add_argument("--voxel-length", type=float, default=0.003)
    parser.add_argument("--voxel-size", type=float, default=0.004)
    parser.add_argument("--sdf-trunc", type=float, default=0.015)
    parser.add_argument("--depth-trunc", type=float, default=5.0)
    parser.add_argument("--depth-tolerance", type=float, default=0.010)
    parser.add_argument("--padding", type=float, default=0.02)
    parser.add_argument("--min-views-checked", type=int, default=2)
    parser.add_argument("--repair-mesh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    from volume_benchmark.bootstrap import ensure_repo_root_on_path

    ensure_repo_root_on_path()
    args = build_parser().parse_args(argv)

    if args.dataset in ("bop_stereo_rendered", "tless_stereo_rendered"):
        if args.dataset_root is None or args.object_id is None:
            print("T-LESS/BOP stereo requires --dataset_root and --object_id", file=sys.stderr)
            return 2
    if args.dataset == "ycb_stereo_rendered" and args.object_root is None:
        print("YCB stereo requires --object_root", file=sys.stderr)
        return 2
    if args.dataset == "wildrgbd_stereo_rendered" and args.prepared_scene_dir is None:
        print("WildRGB-D stereo requires --prepared_scene_dir", file=sys.stderr)
        return 2

    try:
        summary = run_benchmark(args)
        print(f"Wrote {summary}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

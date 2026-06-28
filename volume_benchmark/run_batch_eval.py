"""CLI for batch evaluation over many prepared scans."""

from __future__ import annotations

import argparse
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from volume_benchmark.common.io import validate_prepared_scan
from volume_benchmark.run_eval import METHODS, _run_method, build_parser as eval_parser


def _discover_scans(root: Path) -> list[Path]:
    if (root / "K.npy").is_file() and (root / "gt_volume.json").is_file():
        return [root]
    scans = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "K.npy").is_file():
            scans.append(child)
    if not scans:
        raise FileNotFoundError(
            f"No prepared scans found under {root}. "
            "Expected subdirectories each containing K.npy and gt_volume.json."
        )
    return scans


def build_parser() -> argparse.ArgumentParser:
    base = eval_parser()
    parser = argparse.ArgumentParser(
        description="Batch-evaluate volume methods over many prepared scans.",
        parents=[base],
        conflict_handler="resolve",
    )
    parser.add_argument(
        "scan_root",
        type=Path,
        help="Directory containing prepared scans (or a single scan directory)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory for batch CSV/JSON summaries (default: scan_root/batch_reports)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    for action in list(parser._actions):
        if action.dest == "scan_dir":
            parser._remove_action(action)
            break

    args = parser.parse_args(argv)
    report_dir = args.reports_dir or (args.scan_root / "batch_reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    try:
        scans = _discover_scans(args.scan_root)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    failures = 0

    for scan_dir in tqdm(scans, desc="Scans"):
        if not args.skip_validation:
            errors = validate_prepared_scan(scan_dir)
            if errors:
                failures += 1
                all_rows.append(
                    {"scan_dir": str(scan_dir), "method": None, "error": "; ".join(errors)}
                )
                continue

        for method in args.methods:
            try:
                method_args = argparse.Namespace(**vars(args))
                method_args.output_dir = None
                report = _run_method(method, scan_dir, method_args)
                report["scan_dir"] = str(scan_dir)
                all_rows.append(report)
            except Exception as exc:
                failures += 1
                all_rows.append(
                    {"scan_dir": str(scan_dir), "method": method, "error": str(exc)}
                )

    df = pd.DataFrame(all_rows)
    csv_path = report_dir / "batch_results.csv"
    json_path = report_dir / "batch_results.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    ok = df[df["relative_error_percent"].notna()] if "relative_error_percent" in df.columns else df
    if not ok.empty and "method" in ok.columns:
        summary = (
            ok.groupby("method")["relative_error_percent"]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
        )
        summary.to_csv(report_dir / "summary_by_method.csv", index=False)
        print(summary.to_string(index=False))

    print(f"Wrote {csv_path}, {json_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared helpers for method outputs and GT comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from volume_benchmark.common.io import PreparedScan, load_prepared_scan
from volume_benchmark.common.metrics import relative_error_percent


def resolve_output_dir(scan_dir: str | Path, method_name: str, output_dir: str | Path | None) -> Path:
    scan_path = Path(scan_dir).expanduser().resolve()
    if output_dir is None:
        return scan_path / "outputs" / method_name
    out = Path(output_dir).expanduser().resolve()
    if out.name == method_name:
        return out
    return out / method_name


def gt_comparison_fields(scan: PreparedScan, pred_m3: Optional[float]) -> dict[str, Any]:
    gt_m3 = float(scan.gt_volume["volume_m3"])
    fields: dict[str, Any] = {
        "gt_volume_m3": gt_m3,
        "gt_volume_cm3": gt_m3 * 1e6,
        "gt_type": scan.gt_volume.get("gt_type"),
    }
    if pred_m3 is not None:
        fields["relative_error_percent"] = relative_error_percent(pred_m3, gt_m3)
    else:
        fields["relative_error_percent"] = None
    return fields


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def load_scan_or_raise(scan_dir: str | Path) -> PreparedScan:
    return load_prepared_scan(scan_dir)

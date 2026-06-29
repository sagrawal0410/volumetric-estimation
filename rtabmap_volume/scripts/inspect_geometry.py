"""Inspect input geometry and recommend config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rtabmap_volume.io.load_geometry import inspect_geometry_stats, load_geometry
from rtabmap_volume.preprocess.scale_units import infer_scale_warnings

console = Console()


def recommend_config(stats: dict) -> str:
    dims = stats.get("bbox_dims", [0, 0, 0])
    if len(dims) >= 3 and dims[2] > max(dims[0], dims[1]) * 0.5:
        return "configs/recycling_pile.yaml (tall geometry — pile-like)"
    if stats.get("point_count", stats.get("vertex_count", 0)) > 50000:
        return "configs/fast_preview.yaml (large cloud — preview first)"
    return "configs/object_on_table.yaml (default object on support surface)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inspect RTAB-Map exported geometry")
    p.add_argument("--input", required=True)
    p.add_argument("--units", default="m")
    p.add_argument("--out", default=None, help="Optional output directory for JSON stats")
    args = p.parse_args(argv)

    geom = load_geometry(args.input)
    stats = inspect_geometry_stats(geom)
    diag = stats.get("bbox_diagonal", 0)
    scale_warn = infer_scale_warnings(diag)
    rec = recommend_config(stats)

    table = Table(title="Geometry Inspection")
    table.add_column("Property")
    table.add_column("Value")
    for k, v in stats.items():
        if k != "warnings":
            table.add_row(str(k), str(v))
    console.print(table)

    if scale_warn:
        console.print("[yellow]Scale warnings:[/yellow]")
        for w in scale_warn:
            console.print(f"  - {w}")

    console.print(f"[cyan]Recommended config:[/cyan] {rec}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        payload = {"stats": stats, "scale_warnings": scale_warn, "recommended_config": rec}
        (out / "inspect.json").write_text(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

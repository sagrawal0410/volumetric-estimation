"""HTML evaluation report generator."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _table_from_dict(d: dict[str, Any]) -> str:
    rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in d.items())
    return f"<table border='1' cellpadding='4'>{rows}</table>"


def write_html_report(
    out_path: Path,
    scene_id: str,
    metrics: dict[str, Any],
    volume: dict[str, Any],
    image_paths: dict[str, Path | None],
) -> None:
    imgs = []
    for label, p in image_paths.items():
        if p and Path(p).exists():
            rel = Path(p).name
            imgs.append(f"<div><h4>{html.escape(label)}</h4><img src='{html.escape(rel)}' width='480'/></div>")

    body = f"""
    <html><head><title>Scene {html.escape(scene_id)}</title></head><body>
    <h1>Scene {html.escape(scene_id)}</h1>
    <h2>Metrics</h2>
    {_table_from_dict(metrics)}
    <h2>Volume</h2>
    {_table_from_dict(volume)}
    <h2>Images</h2>
    {''.join(imgs)}
    </body></html>
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")


def write_run_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

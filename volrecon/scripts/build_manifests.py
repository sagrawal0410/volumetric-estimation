"""Build or merge manifest JSONL files from processed scene directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from volrecon.config import DEFAULT_MANIFEST_DIR, DEFAULT_PROCESSED_ROOT, PROJECT_ROOT
from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.io.json_io import read_json, write_jsonl


def _views_from_scene(scene_dir: Path, dataset: str) -> list[dict]:
    scene_meta = read_json(scene_dir / "scene_meta.json")
    scene_id = scene_meta["scene_id"]
    views_dir = scene_dir / "views"
    rows: list[dict] = []
    if not views_dir.exists():
        return rows
    for view_dir in sorted(views_dir.iterdir()):
        if not view_dir.is_dir():
            continue
        meta_path = view_dir / "meta.json"
        meta = read_json(meta_path) if meta_path.exists() else {}
        view_id = view_dir.name
        rec = ViewRecord(
            dataset=dataset,  # type: ignore[arg-type]
            scene_id=scene_id,
            view_id=view_id,
            rgb_path=view_dir / "rgb.png" if (view_dir / "rgb.png").exists() else None,
            left_path=view_dir / "left.png" if (view_dir / "left.png").exists() else None,
            right_path=view_dir / "right.png" if (view_dir / "right.png").exists() else None,
            gt_depth_path=view_dir / "gt_depth.png" if (view_dir / "gt_depth.png").exists() else None,
            notes=meta.get("notes", []),
        )
        rows.append(rec.to_dict(PROJECT_ROOT))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest JSONL from processed directories.")
    parser.add_argument("--processed-root", default=DEFAULT_PROCESSED_ROOT, type=Path)
    parser.add_argument("--dataset", required=True, choices=["robi", "bop_tless"])
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    dataset_dir = args.processed_root / args.dataset
    out = args.out or (DEFAULT_MANIFEST_DIR / f"{args.dataset}_manifest.jsonl")
    rows: list[dict] = []
    for scene_dir in sorted(dataset_dir.iterdir()):
        if scene_dir.is_dir():
            rows.extend(_views_from_scene(scene_dir, args.dataset))
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} view records to {out}")


if __name__ == "__main__":
    main()

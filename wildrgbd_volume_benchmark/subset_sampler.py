"""Create manageable WildRGB-D benchmark subsets."""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path
from typing import Sequence

from wildrgbd_volume_benchmark.io_wildrgbd import discover_scenes, load_types_json


def build_category_scene_manifest(
    wildrgbd_root: str | Path,
    categories: Sequence[str] | None = None,
    output_csv: str | Path | None = None,
) -> Path:
    root = Path(wildrgbd_root).resolve()
    rows: list[dict] = []

    cat_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if categories:
        allowed = {c.lower() for c in categories}
        cat_dirs = [p for p in cat_dirs if p.name.lower() in allowed]

    for cat_dir in cat_dirs:
        types_map = load_types_json(cat_dir)
        scenes_root = cat_dir / "scenes"
        if not scenes_root.is_dir():
            continue
        type_counts = {"single": 0, "multi": 0, "hand": 0, "unknown": 0}
        for scene_path in sorted(scenes_root.iterdir()):
            if not scene_path.is_dir():
                continue
            sid = scene_path.name
            key = sid.replace("scenes_", "")
            stype = types_map.get(key, types_map.get(sid, "unknown"))
            type_counts[stype if stype in type_counts else "unknown"] += 1
            n_rgb = len(list((scene_path / "rgb").glob("*.png"))) if (scene_path / "rgb").is_dir() else 0
            rows.append(
                {
                    "category": cat_dir.name,
                    "scene_id": sid,
                    "scene_type": stype,
                    "num_frames": n_rgb,
                    "scene_dir": str(scene_path),
                }
            )

    out = Path(output_csv) if output_csv else root / "manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return out


def sample_scenes_per_category(
    manifest_csv: str | Path,
    samples_per_category: int = 3,
    scene_types: Sequence[str] = ("single",),
    min_frames: int = 40,
    max_frames: int = 400,
    random_seed: int = 0,
) -> list[dict]:
    manifest_csv = Path(manifest_csv)
    with manifest_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rng = random.Random(random_seed)
    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        if scene_types and row["scene_type"] not in scene_types:
            continue
        nf = int(row.get("num_frames") or 0)
        if nf < min_frames or nf > max_frames:
            continue
        by_cat.setdefault(row["category"], []).append(row)

    selected: list[dict] = []
    for cat, items in sorted(by_cat.items()):
        rng.shuffle(items)
        items.sort(key=lambda r: abs(int(r["num_frames"]) - (min_frames + max_frames) // 2))
        selected.extend(items[:samples_per_category])
    return selected


def materialize_subset(
    wildrgbd_root: str | Path,
    subset_root: str | Path,
    selected_rows: Sequence[dict],
    copy_mode: str = "symlink",
) -> Path:
    root = Path(wildrgbd_root).resolve()
    subset = Path(subset_root).resolve()
    subset.mkdir(parents=True, exist_ok=True)

    cats_touched: set[str] = set()
    for row in selected_rows:
        cat = row["category"]
        cats_touched.add(cat)
        src_scene = Path(row["scene_dir"])
        dst_cat = subset / cat / "scenes" / row["scene_id"]
        dst_cat.parent.mkdir(parents=True, exist_ok=True)
        if dst_cat.exists():
            if dst_cat.is_symlink():
                dst_cat.unlink()
            elif dst_cat.is_dir():
                shutil.rmtree(dst_cat)

        if copy_mode == "symlink":
            dst_cat.symlink_to(src_scene.resolve())
        else:
            shutil.copytree(src_scene, dst_cat)

    for cat in cats_touched:
        cat_src = root / cat
        cat_dst = subset / cat
        for fname in ("types.json", "nvs_list.json", "camera_eval_list.json"):
            src = cat_src / fname
            if src.is_file():
                dst = cat_dst / fname
                if not dst.exists():
                    if copy_mode == "symlink":
                        dst.symlink_to(src.resolve())
                    else:
                        shutil.copy2(src, dst)

    manifest_path = subset / "subset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_rows[0].keys()) if selected_rows else ["category"])
        writer.writeheader()
        writer.writerows(selected_rows)

    summary = {
        "num_scenes": len(selected_rows),
        "categories": sorted(cats_touched),
        "copy_mode": copy_mode,
    }
    with (subset / "subset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return subset

"""Download WildRGB-D categories and extract a small benchmark subset."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from wildrgbd_volume_benchmark.subset_sampler import (
    build_category_scene_manifest,
    materialize_subset,
    sample_scenes_per_category,
)


def _load_categories_from_repo(repo_dir: Path) -> list[str]:
    download_py = repo_dir / "download.py"
    if not download_py.is_file():
        raise FileNotFoundError(
            f"Missing {download_py}. Clone https://github.com/wildrgbd/wildrgbd.git to --repo_dir"
        )
    # Try importing categories from download script
    text = download_py.read_text(encoding="utf-8")
    if "categories" in text and "=" in text:
        import re

        m = re.search(r"categories\s*=\s*\[(.*?)\]", text, re.DOTALL)
        if m:
            items = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
            if items:
                return items
    # Fallback: list category dirs after one download or common set
    return [
        "box", "bottle", "cup", "bowl", "apple", "potato", "shoe", "backpack", "stuffed_toy",
    ]


def _download_category(repo_dir: Path, category: str, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(repo_dir / "download.py"), "--cat", category]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(repo_dir), check=False)
    # WildRGB-D extracts under repo or cwd; search for category folder
    for base in (repo_dir, repo_dir.parent, work_dir, Path.cwd()):
        candidate = base / "WildRGB-D" / category
        if candidate.is_dir():
            return candidate.parent
        candidate = base / category
        if (candidate / "scenes").is_dir():
            wild_root = base if base.name == "WildRGB-D" else base
            return wild_root if wild_root.name == "WildRGB-D" else base
    raise FileNotFoundError(
        f"Could not locate extracted category {category!r} after download. "
        "Check WildRGB-D download.py output location."
    )


def download_subset(
    repo_dir: str | Path,
    work_dir: str | Path,
    subset_root: str | Path,
    samples_per_category: int = 3,
    scene_types: tuple[str, ...] = ("single",),
    categories: list[str] | None = None,
    delete_full_category_after_subset: bool = True,
    copy_mode: str = "symlink",
    min_frames: int = 40,
    max_frames: int = 400,
) -> Path:
    repo_dir = Path(repo_dir).resolve()
    work_dir = Path(work_dir).resolve()
    subset_root = Path(subset_root).resolve()
    subset_root.mkdir(parents=True, exist_ok=True)

    if categories is None or categories == ["all"]:
        cat_list = _load_categories_from_repo(repo_dir)
    else:
        cat_list = categories

    all_selected = []
    log = {"categories": {}, "failures": []}

    for cat in cat_list:
        try:
            wild_root = _download_category(repo_dir, cat, work_dir)
            manifest = build_category_scene_manifest(wild_root, categories=[cat], output_csv=work_dir / f"{cat}_manifest.csv")
            selected = sample_scenes_per_category(
                manifest,
                samples_per_category=samples_per_category,
                scene_types=scene_types,
                min_frames=min_frames,
                max_frames=max_frames,
            )
            if not selected:
                log["failures"].append({"category": cat, "error": "no scenes matched filters"})
                continue
            materialize_subset(wild_root, subset_root, selected, copy_mode=copy_mode)
            all_selected.extend(selected)
            log["categories"][cat] = {"num_scenes": len(selected)}

            if delete_full_category_after_subset:
                cat_full = wild_root / cat if (wild_root / cat).is_dir() else None
                if cat_full and cat_full.resolve() != (subset_root / cat).resolve():
                    shutil.rmtree(cat_full, ignore_errors=True)
                for pat in (f"{cat}.zip", f"{cat}_*.zip", f"*{cat}*.zip"):
                    for z in work_dir.glob(pat):
                        z.unlink(missing_ok=True)
        except Exception as exc:
            log["failures"].append({"category": cat, "error": str(exc)})
            continue

    manifest_path = subset_root / "subset_manifest.csv"
    if all_selected:
        import csv

        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_selected[0].keys()))
            writer.writeheader()
            writer.writerows(all_selected)

    with (subset_root / "subset_summary.json").open("w", encoding="utf-8") as f:
        json.dump({**log, "num_scenes_total": len(all_selected)}, f, indent=2)

    print(f"Subset root: {subset_root} ({len(all_selected)} scenes)")
    return subset_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download WildRGB-D subset by category")
    parser.add_argument("--repo_dir", default="external/wildrgbd")
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--subset_root", default="data/wildrgbd_subset")
    parser.add_argument("--samples_per_category", type=int, default=3)
    parser.add_argument("--scene_types", default="single")
    parser.add_argument("--categories", default="all")
    parser.add_argument("--delete_full_category_after_subset", default="true")
    parser.add_argument("--copy_mode", default="symlink", choices=["symlink", "copy"])
    args = parser.parse_args(argv)

    cats = None if args.categories == "all" else [c.strip() for c in args.categories.split(",")]
    delete = str(args.delete_full_category_after_subset).lower() in ("1", "true", "yes")
    types = tuple(t.strip() for t in args.scene_types.split(",") if t.strip())

    download_subset(
        repo_dir=args.repo_dir,
        work_dir=args.work_dir,
        subset_root=args.subset_root,
        samples_per_category=args.samples_per_category,
        scene_types=types,
        categories=cats,
        delete_full_category_after_subset=delete,
        copy_mode=args.copy_mode,
    )


if __name__ == "__main__":
    main()

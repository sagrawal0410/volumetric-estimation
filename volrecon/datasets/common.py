"""Shared dataset utilities."""

from __future__ import annotations

import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from volrecon.config import PreprocessConfig

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".exr"}

TOKEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "left": re.compile(r"(^|[_\-/\\])(left|l)([_\-/\\]|$|\.)", re.I),
    "right": re.compile(r"(^|[_\-/\\])(right|r)([_\-/\\]|$|\.)", re.I),
    "rgb": re.compile(r"(^|[_\-/\\])(rgb|color|colour)([_\-/\\]|$|\.)", re.I),
    "mono": re.compile(r"(^|[_\-/\\])(mono|gray|grey|ir)([_\-/\\]|$|\.)", re.I),
    "depth": re.compile(r"(^|[_\-/\\])(depth|dep)([_\-/\\]|$|\.)", re.I),
    "gt_depth": re.compile(r"(^|[_\-/\\])(gt[_\-]?depth|ground[_\-]?truth[_\-]?depth)([_\-/\\]|$|\.)", re.I),
    "mask": re.compile(r"(^|[_\-/\\])(mask|seg)([_\-/\\]|$|\.)", re.I),
    "pose": re.compile(r"(^|[_\-/\\])(pose|extrinsic|extrinsics)([_\-/\\]|$|\.)", re.I),
    "camera": re.compile(r"(^|[_\-/\\])(camera|intrinsic|intrinsics|calib)([_\-/\\]|$|\.)", re.I),
    "mesh": re.compile(r"(^|[_\-/\\])(mesh|model|ply|obj)([_\-/\\]|$|\.)", re.I),
}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def classify_file(path: Path) -> list[str]:
    """Classify a file path by filename/folder tokens."""
    text = str(path).lower()
    labels: list[str] = []
    for label, pattern in TOKEN_PATTERNS.items():
        if pattern.search(text):
            labels.append(label)
    if not labels and is_image_file(path):
        labels.append("rgb")
    return labels


def recursive_scan(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if is_image_file(p) or p.suffix.lower() in {".json", ".yaml", ".yml", ".ply", ".obj", ".npz", ".npy", ".txt", ".csv"}:
                files.append(p)
    return sorted(files)


def copy_or_symlink(src: Path, dst: Path, use_symlink: bool = True, overwrite: bool = False) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return dst
        dst.unlink()
    src = src.resolve()
    if use_symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)
    return dst


def extract_view_id(path: Path) -> str:
    stem = path.stem
    m = re.search(r"(\d+)", stem)
    return m.group(1) if m else stem


def group_by_key(paths: Iterable[Path], key_fn) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        groups[key_fn(p)].append(p)
    return dict(groups)


def write_inspection_report(
    out_path: Path,
    dataset_name: str,
    root: Path,
    file_counts: Counter,
    classified: dict[str, list[Path]],
    warnings: list[str],
    notes: list[str],
) -> None:
    lines = [
        f"# Dataset Inspection Report: {dataset_name}",
        "",
        f"Root: `{root}`",
        "",
        "## Summary",
        "",
        f"- Total scanned files: {sum(file_counts.values())}",
        "",
        "### File extensions",
        "",
    ]
    for ext, count in sorted(file_counts.items()):
        lines.append(f"- `{ext}`: {count}")

    lines.extend(["", "## Classification by token", ""])
    for label, paths in sorted(classified.items()):
        lines.append(f"### {label} ({len(paths)})")
        for p in paths[:20]:
            lines.append(f"- `{p.relative_to(root) if p.is_relative_to(root) else p}`")
        if len(paths) > 20:
            lines.append(f"- ... and {len(paths) - 20} more")
        lines.append("")

    if warnings:
        lines.extend(["## Warnings", ""])
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    if notes:
        lines.extend(["## Notes", ""])
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def detect_sensor(path_text: str) -> str:
    text = path_text.lower()
    if "ensenso" in text:
        return "Ensenso"
    if "realsense" in text or "real_sense" in text:
        return "RealSense"
    return "unknown"


def ensure_manifest_dir(cfg: PreprocessConfig, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.processed_root.mkdir(parents=True, exist_ok=True)

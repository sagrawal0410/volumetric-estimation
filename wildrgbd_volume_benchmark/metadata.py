"""Parse WildRGB-D scene metadata (intrinsics, image size)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _parse_K_from_dict(data: dict) -> tuple[np.ndarray, tuple[int, int]]:
    width = int(data.get("width") or data.get("w") or data.get("image_width", 0))
    height = int(data.get("height") or data.get("h") or data.get("image_height", 0))

    if "K" in data:
        K = np.array(data["K"], dtype=np.float64).reshape(3, 3)
    elif "intrinsics" in data:
        K = np.array(data["intrinsics"], dtype=np.float64).reshape(3, 3)
    elif all(k in data for k in ("fx", "fy", "cx", "cy")):
        K = np.array(
            [
                [float(data["fx"]), 0.0, float(data["cx"])],
                [0.0, float(data["fy"]), float(data["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    else:
        raise KeyError(f"No intrinsics found in metadata keys: {list(data.keys())}")

    if width <= 0 or height <= 0:
        width = int(round(2 * K[0, 2]))
        height = int(round(2 * K[1, 2]))
    return K, (width, height)


def _try_parse_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty metadata file")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    try:
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
    except yaml.YAMLError:
        pass

    # key: value lines
    result: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            try:
                if val.startswith("["):
                    result[key] = json.loads(val.replace("'", '"'))
                else:
                    result[key] = float(val) if "." in val else int(val)
            except ValueError:
                result[key] = val
    if result:
        return result

    # fx fy cx cy w h on one line
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", text)]
    if len(nums) >= 4:
        fx, fy, cx, cy = nums[:4]
        w = int(nums[4]) if len(nums) > 4 else int(round(2 * cx))
        h = int(nums[5]) if len(nums) > 5 else int(round(2 * cy))
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": w, "height": h}
    raise ValueError("Could not parse metadata text")


def load_metadata(scene_dir: str | Path) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Parse scene metadata for K and image size.

    Supports a file named ``metadata`` (no extension), ``metadata.json``, or ``metadata.yaml``.
    """
    scene_dir = Path(scene_dir)
    candidates = [
        scene_dir / "metadata",
        scene_dir / "metadata.json",
        scene_dir / "metadata.yaml",
        scene_dir / "metadata.yml",
        scene_dir / "metadata.txt",
    ]
    meta_path = next((p for p in candidates if p.is_file()), None)
    if meta_path is None:
        raise FileNotFoundError(
            f"No metadata file found in {scene_dir}. Expected one of: "
            + ", ".join(p.name for p in candidates)
        )

    text = meta_path.read_text(encoding="utf-8", errors="replace")
    try:
        data = _try_parse_text(text)
        return _parse_K_from_dict(data)
    except Exception as exc:
        sample = text[:500].replace("\n", "\\n")
        raise ValueError(
            f"Failed to parse metadata at {meta_path}: {exc}\n"
            f"File sample (first 500 chars): {sample}"
        ) from exc

"""Image I/O utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def read_rgb(path: Path) -> np.ndarray:
    img = read_image(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3 and image.shape[2] == 3:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), image)


def read_depth_png(path: Path, scale: float = 1.0) -> np.ndarray:
    """Read a depth PNG (typically uint16) and scale to meters."""
    depth = read_image(path, cv2.IMREAD_UNCHANGED)
    return depth.astype(np.float64) * scale


def read_mask(path: Path) -> np.ndarray:
    mask = read_image(path, cv2.IMREAD_GRAYSCALE)
    return mask > 0


def image_shape(path: Path) -> tuple[int, int]:
    img = read_image(path, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    return w, h

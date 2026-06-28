"""Mask loading utilities for WildRGB-D."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_mask(mask_path: str | Path) -> np.ndarray:
    """Load mask PNG as boolean; values >127 are foreground."""
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return raw > 127

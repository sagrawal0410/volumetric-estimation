"""Shared stereo types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DepthPrediction:
    disparity_px: np.ndarray
    depth_m: np.ndarray
    valid_mask: np.ndarray
    K: np.ndarray
    baseline_m: float
    meta: dict[str, Any]

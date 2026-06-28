"""Robust weighting kernels for TSDF updates."""

from __future__ import annotations

import numpy as np


def huber_weight(residual: np.ndarray, delta: float) -> np.ndarray:
    r = np.abs(np.asarray(residual, dtype=np.float64))
    d = float(delta)
    w = np.ones_like(r)
    mask = r > d
    w[mask] = d / np.maximum(r[mask], 1e-12)
    return w


def tukey_weight(residual: np.ndarray, c: float) -> np.ndarray:
    r = np.abs(np.asarray(residual, dtype=np.float64))
    c = float(c)
    w = np.zeros_like(r)
    mask = r <= c
    w[mask] = (1.0 - (r[mask] / c) ** 2) ** 2
    return w

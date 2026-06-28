"""Rigid transform utilities."""

from __future__ import annotations

import numpy as np


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return make_T(R_inv, t_inv)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hom = np.hstack([pts, ones])
    out = (T @ hom.T).T
    return out[:, :3]


def bop_T_model_cam_to_meters(R_m2c: list[float], t_m2c_mm: list[float]) -> np.ndarray:
    """Build T_model_cam from BOP pose fields (translation in mm -> meters)."""
    R = np.asarray(R_m2c, dtype=np.float64).reshape(3, 3)
    t_m = np.asarray(t_m2c_mm, dtype=np.float64).reshape(3) * 0.001
    return make_T(R, t_m)


def bop_T_cam_model_to_meters(R_m2c: list[float], t_m2c_mm: list[float]) -> np.ndarray:
    return invert_T(bop_T_model_cam_to_meters(R_m2c, t_m2c_mm))


def ensure_right_handed(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    if np.linalg.det(R) < 0:
        R = -R
    return R


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    if degrees:
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

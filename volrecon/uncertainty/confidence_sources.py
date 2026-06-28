"""Per-pixel confidence component computation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from volrecon.uncertainty.calibration import UncertaintyConfig
from volrecon.uncertainty.stereo_consistency import (
    lr_consistency_confidence,
    photometric_warp_right_to_left,
)


@dataclass
class ConfidenceMaps:
    valid: np.ndarray
    c_lr: np.ndarray
    c_photo: np.ndarray
    c_range: np.ndarray
    c_angle: np.ndarray
    c_texture: np.ndarray
    c_sat: np.ndarray
    c_mv: np.ndarray
    c_temp: np.ndarray
    confidence_total: np.ndarray
    weight_total: np.ndarray


def disparity_validity(disparity: np.ndarray, min_disp: float) -> np.ndarray:
    d = np.asarray(disparity, dtype=np.float64)
    return np.isfinite(d) & (d > min_disp)


def range_weight_from_depth(depth_m: np.ndarray, sigma_min: float = 0.001, k_z: float = 1e-4) -> np.ndarray:
    z = np.asarray(depth_m, dtype=np.float64)
    sigma_z = sigma_min + k_z * z**2
    w = 1.0 / (sigma_z**2 + 1e-12)
    valid = z > 0
    if np.any(valid):
        w_norm = w.copy()
        w_norm[valid] = w[valid] / np.percentile(w[valid], 95)
        w_norm = np.clip(w_norm, 0.0, 1.0)
        return w_norm
    return np.zeros_like(z)


def normals_from_depth(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Estimate camera-frame normals from depth map."""
    z = np.asarray(depth_m, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) * z / fx
    y = (v - K[1, 2]) * z / fy
    pts = np.stack([x, y, z], axis=-1)

    dx = np.zeros_like(pts)
    dy = np.zeros_like(pts)
    dx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dy[1:-1, :] = pts[2:, :] - pts[:-2, :]
    normals = np.cross(dx, dy)
    nrm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(nrm, 1e-12)
    return normals


def view_angle_weight(depth_m: np.ndarray, K: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    normals = normals_from_depth(depth_m, K)
    view_dir = np.zeros_like(normals)
    view_dir[..., 2] = 1.0  # camera looks along +Z in camera frame
    dot = np.sum(normals * view_dir, axis=-1)
    c = np.clip(dot, 0.0, 1.0) ** gamma
    c[depth_m <= 0] = 0.0
    return c


def texture_weight(left_img: np.ndarray, low_thresh: float = 0.01, scale: float = 0.05) -> np.ndarray:
    if left_img.ndim == 3:
        gray = cv2.cvtColor(left_img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    else:
        gray = left_img.astype(np.float64)
    local_var = cv2.GaussianBlur(gray**2, (5, 5), 0) - cv2.GaussianBlur(gray, (5, 5), 0) ** 2
    local_var = np.maximum(local_var, 0.0)
    c = 1.0 / (1.0 + np.exp(-(local_var - low_thresh) / max(scale, 1e-6)))
    return np.clip(c, 0.0, 1.0)


def saturation_weight(img: np.ndarray, low: float = 0.02, high: float = 0.98) -> np.ndarray:
    if img.ndim == 3:
        gray = img.astype(np.float64)
        if gray.max() > 1.0:
            gray = gray / 255.0
        intensity = gray.max(axis=-1)
    else:
        intensity = img.astype(np.float64)
        if intensity.max() > 1.0:
            intensity = intensity / 255.0
    c = np.ones_like(intensity)
    bad = (intensity < low) | (intensity > high)
    c[bad] = 0.2
    return c


def photometric_considence(left_img: np.ndarray, right_img: np.ndarray, disparity: np.ndarray, tau: float) -> np.ndarray:
    warped, valid = photometric_warp_right_to_left(right_img, disparity)
    left = left_img.astype(np.float64)
    if left.max() > 1.0:
        left = left / 255.0
    if warped.max() > 1.0:
        warped = warped / 255.0
    err = np.mean(np.abs(left - warped), axis=-1) if left.ndim == 3 else np.abs(left - warped)
    c = np.exp(-err / max(tau, 1e-6))
    c[~valid] = 0.0
    return np.clip(c, 0.0, 1.0)


def combine_confidence(components: dict[str, np.ndarray], cfg: UncertaintyConfig) -> tuple[np.ndarray, np.ndarray]:
    exp = cfg.exponents
    wmap = cfg.weights
    valid = components.get("valid", np.ones_like(next(iter(components.values()))))
    c = valid.astype(np.float64)
    for key, alpha in [
        ("c_lr", exp.alpha_lr),
        ("c_photo", exp.alpha_photo),
        ("c_range", exp.alpha_range),
        ("c_angle", exp.alpha_angle),
        ("c_texture", exp.alpha_texture),
        ("c_sat", exp.alpha_sat),
        ("c_mv", exp.alpha_mv),
        ("c_temp", exp.alpha_temp),
    ]:
        comp = components.get(key)
        if comp is not None:
            c *= np.power(np.clip(comp, 0.0, 1.0), alpha)
    c = np.clip(c, 0.0, 1.0)
    weight = wmap.w_min + wmap.w_scale * c
    weight = np.clip(weight, 0.0, wmap.w_max_per_obs)
    weight[~valid] = 0.0
    return c, weight


def compute_confidence_maps(
    depth_m: np.ndarray,
    disparity: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    K: np.ndarray,
    cfg: UncertaintyConfig,
    disparity_r2l: np.ndarray | None = None,
    c_mv: np.ndarray | None = None,
    c_temp: np.ndarray | None = None,
) -> ConfidenceMaps:
    thr = cfg.thresholds
    valid = disparity_validity(disparity, thr.min_disp) & (depth_m > 0)
    c_lr = lr_consistency_confidence(disparity, disparity_r2l, thr.tau_lr_px)
    c_photo = photometric_confidence(left_img, right_img, disparity, thr.tau_photo)
    c_range = range_weight_from_depth(depth_m, thr.sigma_min, thr.k_z)
    c_angle = view_angle_weight(depth_m, K, thr.view_angle_gamma)
    c_texture = texture_weight(left_img, thr.texture_low, thr.texture_scale)
    c_sat = saturation_weight(left_img)
    if c_mv is None:
        c_mv = np.ones_like(depth_m)
    if c_temp is None:
        c_temp = np.ones_like(depth_m)

    components = {
        "valid": valid,
        "c_lr": c_lr,
        "c_photo": c_photo,
        "c_range": c_range,
        "c_angle": c_angle,
        "c_texture": c_texture,
        "c_sat": c_sat,
        "c_mv": c_mv,
        "c_temp": c_temp,
    }
    c_total, w_total = combine_confidence(components, cfg)
    return ConfidenceMaps(
        valid=valid.astype(np.float32),
        c_lr=c_lr.astype(np.float32),
        c_photo=c_photo.astype(np.float32),
        c_range=c_range.astype(np.float32),
        c_angle=c_angle.astype(np.float32),
        c_texture=c_texture.astype(np.float32),
        c_sat=c_sat.astype(np.float32),
        c_mv=c_mv.astype(np.float32),
        c_temp=c_temp.astype(np.float32),
        confidence_total=c_total.astype(np.float32),
        weight_total=w_total.astype(np.float32),
    )


def save_confidence_maps(out_dir: Path, maps: ConfidenceMaps) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "confidence_total.npy", maps.confidence_total)
    np.save(out_dir / "weight_total.npy", maps.weight_total)
    np.save(out_dir / "stereo_lr_consistency.npy", maps.c_lr)
    np.save(out_dir / "photometric_consistency.npy", maps.c_photo)
    np.save(out_dir / "range_weight.npy", maps.c_range)
    np.save(out_dir / "view_angle_weight.npy", maps.c_angle)
    np.save(out_dir / "texture_weight.npy", maps.c_texture)
    np.save(out_dir / "saturation_weight.npy", maps.c_sat)
    np.save(out_dir / "multiview_agreement.npy", maps.c_mv)
    np.save(out_dir / "temporal_stability.npy", maps.c_temp)
    cm = (maps.confidence_total * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "confidence_debug.png"), cv2.applyColorMap(cm, cv2.COLORMAP_VIRIDIS))

"""Uncertainty weight calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UncertaintyExponents:
    alpha_lr: float = 2.0
    alpha_photo: float = 1.0
    alpha_range: float = 1.0
    alpha_angle: float = 1.0
    alpha_texture: float = 0.5
    alpha_sat: float = 1.0
    alpha_mv: float = 2.0
    alpha_temp: float = 1.0


@dataclass
class UncertaintyThresholds:
    tau_lr_px: float = 1.5
    tau_photo: float = 0.08
    tau_mv_m: float = 0.005
    min_disp: float = 0.5
    sigma_min: float = 0.001
    k_z: float = 1e-4
    texture_low: float = 0.01
    texture_scale: float = 0.05
    view_angle_gamma: float = 1.0


@dataclass
class WeightMapping:
    w_min: float = 0.01
    w_scale: float = 5.0
    w_max_per_obs: float = 5.0


@dataclass
class RobustKernelConfig:
    type: str = "huber"
    delta: float = 0.25


@dataclass
class UncertaintyConfig:
    exponents: UncertaintyExponents = field(default_factory=UncertaintyExponents)
    thresholds: UncertaintyThresholds = field(default_factory=UncertaintyThresholds)
    weights: WeightMapping = field(default_factory=WeightMapping)
    robust: RobustKernelConfig = field(default_factory=RobustKernelConfig)
    run_right_to_left: bool = True
    k_neighbor_views: int = 5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UncertaintyConfig":
        unc = d.get("uncertainty", d)
        exp = unc.get("exponents", unc)
        thr = unc.get("thresholds", unc)
        wgt = unc.get("weights", unc)
        rob = unc.get("robust_kernel", d.get("robust_kernel", {}))
        return cls(
            exponents=UncertaintyExponents(
                alpha_lr=exp.get("alpha_lr", 2.0),
                alpha_photo=exp.get("alpha_photo", 1.0),
                alpha_range=exp.get("alpha_range", 1.0),
                alpha_angle=exp.get("alpha_angle", 1.0),
                alpha_texture=exp.get("alpha_texture", 0.5),
                alpha_sat=exp.get("alpha_sat", 1.0),
                alpha_mv=exp.get("alpha_mv", 2.0),
                alpha_temp=exp.get("alpha_temp", 1.0),
            ),
            thresholds=UncertaintyThresholds(
                tau_lr_px=unc.get("tau_lr_px", thr.get("tau_lr_px", 1.5)),
                tau_photo=unc.get("tau_photo", thr.get("tau_photo", 0.08)),
                tau_mv_m=unc.get("tau_mv_m", thr.get("tau_mv_m", 0.005)),
                min_disp=thr.get("min_disp", 0.5),
            ),
            weights=WeightMapping(
                w_min=unc.get("w_min", wgt.get("w_min", 0.01)),
                w_scale=unc.get("w_scale", wgt.get("w_scale", 5.0)),
                w_max_per_obs=unc.get("w_max_per_obs", wgt.get("w_max_per_obs", 5.0)),
            ),
            robust=RobustKernelConfig(
                type=rob.get("type", "huber"),
                delta=rob.get("delta", 0.25),
            ),
            run_right_to_left=unc.get("run_right_to_left", True),
            k_neighbor_views=unc.get("k_neighbor_views", 5),
        )

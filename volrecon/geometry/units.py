"""Unit conversion utilities."""

from __future__ import annotations

UNIT_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
}


def unit_scale_to_meters(unit: str) -> float:
    key = unit.strip().lower()
    if key not in UNIT_TO_METERS:
        raise ValueError(f"Unknown unit: {unit!r}. Supported: {sorted(UNIT_TO_METERS)}")
    return UNIT_TO_METERS[key]


def convert_length(value: float, from_unit: str, to_unit: str = "m") -> float:
    meters = value * unit_scale_to_meters(from_unit)
    return meters / unit_scale_to_meters(to_unit)


def mm_to_m(value: float | list[float]) -> float | list[float]:
    if isinstance(value, list):
        return [v * 0.001 for v in value]
    return value * 0.001

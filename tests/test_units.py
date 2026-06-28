"""Tests for unit conversion."""

from __future__ import annotations

import numpy as np

from volrecon.geometry.transforms import bop_T_model_cam_to_meters, make_T
from volrecon.geometry.units import convert_length, mm_to_m, unit_scale_to_meters


def test_mm_to_m_scalar():
    assert mm_to_m(1000.0) == 1.0


def test_mm_to_m_list():
    assert mm_to_m([1000.0, 2000.0]) == [1.0, 2.0]


def test_convert_length():
    assert convert_length(1000.0, "mm", "m") == 1.0


def test_bop_translation_converted_to_meters():
    T = bop_T_model_cam_to_meters([1, 0, 0, 0, 1, 0, 0, 0, 1], [1000.0, 0.0, 0.0])
    assert np.isclose(T[0, 3], 1.0)


def test_unit_scale_to_meters():
    assert unit_scale_to_meters("mm") == 0.001

"""Tests for known analytical volumes."""

import numpy as np
import trimesh

from rtabmap_volume.eval.synthetic_tests import analytical_volume, make_cube, make_cylinder, make_sphere
from rtabmap_volume.volume.mesh_volume import compute_mesh_volume


def test_cube_volume():
    cube = make_cube(1.0)
    est = compute_mesh_volume(cube)
    assert est.value_m3 is not None
    assert abs(est.value_m3 - 1.0) < 0.02
    assert est.reliable


def test_sphere_volume():
    sphere = make_sphere(0.5, subdivisions=4)
    est = compute_mesh_volume(sphere)
    truth = analytical_volume("sphere", radius=0.5)
    assert est.value_m3 is not None
    assert abs(est.value_m3 - truth) / truth < 0.05


def test_cylinder_volume():
    cyl = make_cylinder(radius=0.3, height=1.0)
    est = compute_mesh_volume(cyl)
    truth = analytical_volume("cylinder", radius=0.3, height=1.0)
    assert est.value_m3 is not None
    assert abs(est.value_m3 - truth) / truth < 0.05

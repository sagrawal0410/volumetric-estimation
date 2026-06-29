"""Mesh volume reliability tests."""

import trimesh

from rtabmap_volume.eval.synthetic_tests import make_partial_cube
from rtabmap_volume.volume.mesh_volume import compute_mesh_volume


def test_non_watertight_not_high_confidence():
    partial = make_partial_cube()
    est = compute_mesh_volume(partial)
    assert not est.reliable

"""Consensus selection tests."""

import trimesh

from rtabmap_volume.config import ConsensusConfig
from rtabmap_volume.eval.synthetic_tests import make_cube
from rtabmap_volume.volume.consensus import compute_consensus
from rtabmap_volume.volume.mesh_volume import VolumeEstimate, compute_mesh_volume
from rtabmap_volume.volume.voxel_volume import compute_voxel_volumes
import open3d as o3d
import numpy as np


def test_watertight_mesh_consensus():
    cube = make_cube(1.0)
    direct = compute_mesh_volume(cube)
    estimates = {"direct_mesh_volume": direct}
    result = compute_consensus(estimates, ConsensusConfig())
    assert result.recommended_estimator == "direct_mesh_volume"
    assert result.confidence == "high"
    assert abs(result.final_volume_m3 - 1.0) < 0.05


def test_pile_config_prefers_heightfield():
    estimates = {
        "heightfield_volume": VolumeEstimate("heightfield_volume", 2.5, 2500, True, []),
        "voxel_occupancy_volume": VolumeEstimate("voxel_occupancy_volume", 2.3, 2300, True, []),
        "direct_mesh_volume": VolumeEstimate("direct_mesh_volume", None, None, False, ["not watertight"]),
    }
    cfg = ConsensusConfig(
        pile_mode=True,
        estimator_priority=["heightfield_volume", "voxel_occupancy_volume", "direct_mesh_volume"],
    )
    result = compute_consensus(estimates, cfg)
    assert result.recommended_estimator == "heightfield_volume"
    assert result.final_volume_m3 == 2.5

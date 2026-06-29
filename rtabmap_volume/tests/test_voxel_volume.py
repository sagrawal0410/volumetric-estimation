"""Voxel volume convergence tests."""

import numpy as np
import open3d as o3d
import trimesh

from rtabmap_volume.config import VoxelConfig
from rtabmap_volume.eval.synthetic_tests import make_cube
from rtabmap_volume.volume.voxel_volume import compute_voxel_volumes


def test_voxel_cube_converges():
    cube = make_cube(1.0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(cube.vertices))
    est_coarse, _ = compute_voxel_volumes(pcd=pcd, mesh=cube, cfg=VoxelConfig(voxel_sizes_m=[0.1]))
    est_fine, _ = compute_voxel_volumes(pcd=pcd, mesh=cube, cfg=VoxelConfig(voxel_sizes_m=[0.02]))
    assert est_coarse.value_m3 is not None
    assert est_fine.value_m3 is not None
    assert abs(est_fine.value_m3 - 1.0) < abs(est_coarse.value_m3 - 1.0)

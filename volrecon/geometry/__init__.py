"""Geometry subpackage."""

from volrecon.geometry.camera import (
    CameraIntrinsics,
    K_to_intrinsics,
    backproject_depth,
    depth_to_pointcloud,
    disparity_to_depth,
    project_points,
    resize_intrinsics,
    validate_positive_depth,
)
from volrecon.geometry.depth import apply_depth_scale, depth_uint16_to_meters
from volrecon.geometry.mesh_volume import mesh_volume_m3, union_voxel_grids, voxelize_mesh
from volrecon.geometry.transforms import (
    bop_T_cam_model_to_meters,
    bop_T_model_cam_to_meters,
    ensure_right_handed,
    invert_T,
    make_T,
    transform_points,
)
from volrecon.geometry.units import convert_length, mm_to_m, unit_scale_to_meters

__all__ = [
    "CameraIntrinsics",
    "K_to_intrinsics",
    "backproject_depth",
    "depth_to_pointcloud",
    "disparity_to_depth",
    "project_points",
    "resize_intrinsics",
    "validate_positive_depth",
    "apply_depth_scale",
    "depth_uint16_to_meters",
    "mesh_volume_m3",
    "union_voxel_grids",
    "voxelize_mesh",
    "bop_T_cam_model_to_meters",
    "bop_T_model_cam_to_meters",
    "ensure_right_handed",
    "invert_T",
    "make_T",
    "transform_points",
    "convert_length",
    "mm_to_m",
    "unit_scale_to_meters",
]

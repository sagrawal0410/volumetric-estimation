"""TSDF fusion utilities and pose conventions."""

from __future__ import annotations

import numpy as np

from volrecon.datasets.canonical_schema import ViewRecord
from volrecon.geometry.transforms import invert_T


class PoseConventionError(ValueError):
    pass


def extrinsic_for_open3d(T_world_cam: np.ndarray | None, T_cam_world: np.ndarray | None) -> np.ndarray:
    """
    Open3D TSDF expects extrinsic mapping world -> camera (T_cam_world).

    Internal convention:
      T_world_cam: p_world = T_world_cam @ p_cam
      T_cam_world = inverse(T_world_cam)
    """
    if T_cam_world is not None:
        T = np.asarray(T_cam_world, dtype=np.float64).reshape(4, 4)
        return T
    if T_world_cam is not None:
        return invert_T(np.asarray(T_world_cam, dtype=np.float64).reshape(4, 4))
    raise PoseConventionError("Need T_world_cam or T_cam_world for Open3D integration")


def world_cam_from_view(view: ViewRecord, object_centric: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve (T_world_cam, T_cam_world) for fusion.

    When object_centric=True (BOP without world pose), world frame = model frame of first object:
      T_world_cam = T_cam_model (maps camera points to model/world frame)
      T_cam_world = T_model_cam
    """
    if view.T_world_cam is not None:
        T_wc = np.asarray(view.T_world_cam, dtype=np.float64).reshape(4, 4)
        T_cw = (
            np.asarray(view.T_cam_world, dtype=np.float64).reshape(4, 4)
            if view.T_cam_world is not None
            else invert_T(T_wc)
        )
        return T_wc, T_cw

    if object_centric and view.object_poses:
        op = view.object_poses[0]
        T_model_cam = np.asarray(op.T_model_cam, dtype=np.float64).reshape(4, 4)
        T_cam_model = np.asarray(op.T_cam_model, dtype=np.float64).reshape(4, 4)
        # world = model frame
        T_wc = T_cam_model
        T_cw = T_model_cam
        return T_wc, T_cw

    raise PoseConventionError(
        f"No camera pose for {view.scene_id}/{view.view_id}. "
        "ROBI requires T_world_cam. BOP cluttered scenes need cam_R_w2c/cam_t_w2c or object-centric mode."
    )


def open3d_intrinsic(K: np.ndarray, width: int, height: int):
    import open3d as o3d

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    return o3d.camera.PinholeCameraIntrinsic(
        width=int(width),
        height=int(height),
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )

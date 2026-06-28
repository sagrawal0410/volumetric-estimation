"""Ground-truth mesh rendering for BOP synthetic stereo and GT depth."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import trimesh

from volrecon.geometry.camera import project_points
from volrecon.geometry.transforms import invert_T, make_T, transform_points
from volrecon.io.mesh_io import load_mesh

logger = logging.getLogger(__name__)

_HEADLESS_WARNED = False
_USE_PYRENDER = os.environ.get("VOLRECON_USE_PYRENDER", "").lower() in {"1", "true", "yes"}


def _warn_headless_once(msg: str) -> None:
    global _HEADLESS_WARNED
    if not _HEADLESS_WARNED:
        logger.warning(msg)
        _HEADLESS_WARNED = True


def _try_import_pyrender():
    try:
        import pyrender  # noqa: WPS433

        return pyrender
    except ImportError:
        return None


def render_mesh_depth(
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    width: int,
    height: int,
    T_cam_model: np.ndarray | None = None,
) -> np.ndarray:
    """Render a depth map (meters) from a mesh in camera frame."""
    if not _USE_PYRENDER:
        return _raycast_depth_fallback(mesh, K, width, height, T_cam_model)

    pyrender = _try_import_pyrender()
    if pyrender is None:
        _warn_headless_once("pyrender unavailable; using ray-based depth fallback")
        return _raycast_depth_fallback(mesh, K, width, height, T_cam_model)

    try:
        return _render_depth_pyrender(mesh, K, width, height, T_cam_model, pyrender)
    except Exception as exc:  # noqa: BLE001
        _warn_headless_once(f"pyrender failed ({exc}); using ray-based depth fallback")
        return _raycast_depth_fallback(mesh, K, width, height, T_cam_model)


def _render_depth_pyrender(mesh, K, width, height, T_cam_model, pyrender):
    import pyrender as pr

    m = mesh.copy()
    if T_cam_model is not None:
        m.apply_transform(invert_T(T_cam_model))

    scene = pr.Scene(ambient_light=[0.5, 0.5, 0.5])
    pr_mesh = pr.Mesh.from_trimesh(m, smooth=False)
    scene.add(pr_mesh)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    camera = pr.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=10.0)
    camera_pose = np.eye(4)
    scene.add(camera, pose=camera_pose)

    renderer = pr.OffscreenRenderer(width, height)
    try:
        _, depth = renderer.render(scene)
    finally:
        renderer.delete()
    depth = depth.astype(np.float64)
    depth[depth <= 0] = 0.0
    return depth


def _raycast_depth_fallback(
    mesh: trimesh.Trimesh,
    K: np.ndarray,
    width: int,
    height: int,
    T_cam_model: np.ndarray | None,
) -> np.ndarray:
    """Z-buffer triangle rasterization fallback (headless-safe)."""
    m = mesh.copy()
    if T_cam_model is not None:
        m.apply_transform(invert_T(T_cam_model))
    verts_cam = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.int64)
    depth = np.full((height, width), np.inf, dtype=np.float64)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    for tri in faces:
        pts = verts_cam[tri]
        if np.any(pts[:, 2] <= 1e-6):
            continue
        u = fx * pts[:, 0] / pts[:, 2] + cx
        v = fy * pts[:, 1] / pts[:, 2] + cy
        u_min = max(int(np.floor(u.min())), 0)
        u_max = min(int(np.ceil(u.max())), width - 1)
        v_min = max(int(np.floor(v.min())), 0)
        v_max = min(int(np.ceil(v.max())), height - 1)
        if u_min > u_max or v_min > v_max:
            continue

        p0, p1, p2 = pts
        denom2 = (v[1] - v[2]) * (u[0] - u[2]) + (u[2] - u[1]) * (v[0] - v[2])
        if abs(denom2) < 1e-12:
            continue
        normal = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(normal) < 1e-12:
            continue

        for vv in range(v_min, v_max + 1):
            for uu in range(u_min, u_max + 1):
                w0 = ((v[1] - v[2]) * (uu - u[2]) + (u[2] - u[1]) * (vv - v[2])) / denom2
                w1 = ((v[2] - v[0]) * (uu - u[2]) + (u[0] - u[2]) * (vv - v[2])) / denom2
                w2 = 1.0 - w0 - w1
                if w0 < -1e-4 or w1 < -1e-4 or w2 < -1e-4:
                    continue
                z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
                if z <= 0 or z >= depth[vv, uu]:
                    continue
                depth[vv, uu] = z

    out = np.zeros((height, width), dtype=np.float64)
    valid = np.isfinite(depth)
    out[valid] = depth[valid]
    return out


def render_synthetic_stereo_pair(
    meshes: list[trimesh.Trimesh],
    poses_cam_model: list[np.ndarray],
    K: np.ndarray,
    width: int,
    height: int,
    baseline_m: float,
) -> dict[str, np.ndarray | float | bool]:
    """Render synthetic left/right from original camera + baseline shift along +X."""
    combined = trimesh.util.concatenate(
        [
            m.copy().apply_transform(invert_T(T))
            for m, T in zip(meshes, poses_cam_model, strict=True)
        ]
    )
    left_depth = render_mesh_depth(combined, K, width, height, T_cam_model=None)

    T_left_right = make_T(np.eye(3), np.array([baseline_m, 0.0, 0.0]))
    T_right_left = invert_T(T_left_right)
    combined_right = combined.copy()
    combined_right.apply_transform(T_right_left)
    right_depth = render_mesh_depth(combined_right, K, width, height, T_cam_model=None)

    left_rgb = _depth_to_gray_rgb(left_depth)
    right_rgb = _depth_to_gray_rgb(right_depth)

    return {
        "left_rgb": left_rgb,
        "right_rgb": right_rgb,
        "left_depth_m": left_depth,
        "right_depth_m": right_depth,
        "baseline_m": baseline_m,
        "synthetic": True,
        "T_left_right": T_left_right,
    }


def _depth_to_gray_rgb(depth_m: np.ndarray) -> np.ndarray:
    valid = depth_m > 0
    rgb = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return rgb
    d = depth_m.copy()
    ref = float(np.percentile(d[valid], 95))
    ref = max(ref, 1e-3)
    gray = np.clip(d / ref * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 0] = gray
    rgb[..., 1] = gray
    rgb[..., 2] = gray
    rgb[~valid] = 0
    return rgb


def transform_mesh_to_scene(mesh: trimesh.Trimesh, T_scene_model: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    out.apply_transform(T_scene_model)
    return out


def render_scene_gt_depth_from_objects(
    object_meshes: dict[int, trimesh.Trimesh],
    object_poses_cam_model: dict[int, np.ndarray],
    K: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    meshes = []
    poses = []
    for obj_id, mesh in object_meshes.items():
        if obj_id not in object_poses_cam_model:
            continue
        meshes.append(mesh)
        poses.append(object_poses_cam_model[obj_id])
    if not meshes:
        return np.zeros((height, width), dtype=np.float64)
    combined = trimesh.util.concatenate(
        [m.copy().apply_transform(invert_T(T)) for m, T in zip(meshes, poses, strict=True)]
    )
    return render_mesh_depth(combined, K, width, height)


def load_and_transform_bop_model(model_path: Path, T_scene_model: np.ndarray, mm_to_m: bool = True) -> trimesh.Trimesh:
    mesh = load_mesh(model_path)
    if mm_to_m:
        mesh.apply_scale(0.001)
    return transform_mesh_to_scene(mesh, T_scene_model)

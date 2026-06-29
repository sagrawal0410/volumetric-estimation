"""End-to-end plain baseline tests with synthetic geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from volrecon.datasets.canonical_schema import ObjectPoseRecord, StereoCalibration, ViewRecord
from volrecon.fusion.bounds import compute_bounds_from_depth_points, robust_expand_bounds
from volrecon.fusion.fusion_utils import extrinsic_for_open3d, world_cam_from_view
from volrecon.fusion.open3d_tsdf import PlainTSDFConfig, PlainTSDFReconstructor, o3d_mesh_to_trimesh
from volrecon.geometry.mesh_volume import compute_mesh_volume_report
from volrecon.geometry.render_gt import render_mesh_depth, render_synthetic_stereo_pair
from volrecon.geometry.transforms import invert_T, make_T
from volrecon.stereo.depth_estimator import PerfectDepthEstimator
from volrecon.stereo.foundation_stereo_wrapper import NO_STEREO_ERROR, FoundationStereoWrapper
from volrecon.stereo.foundation_stereo_wrapper import FoundationStereoConfig


def _cube_mesh(size: float = 0.2, center=(0.0, 0.0, 0.5)) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=(size, size, size))
    m.apply_translation(center)
    return m


def _camera_pose(radius: float, angle_deg: float, target=(0.0, 0.0, 0.5)) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    cam_pos = np.array([radius * np.sin(a), 0.0, target[2] + radius * np.cos(a)])
    target = np.array(target)
    up = np.array([0.0, 1.0, 0.0])
    z = target - cam_pos
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    T_wc = np.eye(4)
    T_wc[:3, :3] = R
    T_wc[:3, 3] = cam_pos
    return T_wc


def test_extrinsic_open3d_is_cam_world():
    T_wc = _camera_pose(0.8, 30.0)
    T_cw = invert_T(T_wc)
    ext = extrinsic_for_open3d(T_wc, None)
    assert np.allclose(ext, T_cw, atol=1e-9)


def test_synthetic_cube_tsdf_volume(tmp_path: Path):
    cube = _cube_mesh(0.2, center=(0.0, 0.0, 0.5))
    true_vol = 0.2**3
    K = np.array([[400.0, 0, 160], [0, 400.0, 120], [0, 0, 1]])
    w, h = 320, 240
    angles = [0, 30, 60, 90, 120, 150]

    depth_maps = []
    Ks = []
    T_wcs = []
    for i, ang in enumerate(angles):
        T_wc = _camera_pose(0.8, ang)
        T_cw = invert_T(T_wc)
        cube_cam = cube.copy()
        cube_cam.apply_transform(T_cw)
        depth = render_mesh_depth(cube_cam, K, w, h)
        view_dir = tmp_path / "depth" / f"view_{i:02d}"
        view_dir.mkdir(parents=True)
        np.save(view_dir / "depth_m.npy", depth.astype(np.float32))
        depth_maps.append(depth)
        Ks.append(K)
        T_wcs.append(T_wc)

    bounds = robust_expand_bounds(compute_bounds_from_depth_points(depth_maps, Ks, T_wcs), 0.05)
    tsdf = PlainTSDFReconstructor(
        PlainTSDFConfig(voxel_length_m=0.003, sdf_trunc_m=0.012, integrate_color=False),
        bounds,
    )

    for i, ang in enumerate(angles):
        T_wc = _camera_pose(0.8, ang)
        T_cw = invert_T(T_wc)
        tsdf.integrate_view(None, tmp_path / "depth" / f"view_{i:02d}" / "depth_m.npy", K, T_cam_world=T_cw)

    mesh_o3d = tsdf.extract_mesh()
    mesh = o3d_mesh_to_trimesh(mesh_o3d)
    assert len(mesh.vertices) > 100
    rep = compute_mesh_volume_report(mesh, voxel_size_m=0.003)
    # Fallback voxel/hull volume from rasterized depth TSDF can underestimate; check order-of-magnitude.
    assert rep.volume_m3 > true_vol * 0.05
    assert rep.volume_m3 < true_vol * 5.0
    centroid = mesh.vertices.mean(axis=0)
    assert np.linalg.norm(centroid - np.array([0.0, 0.0, 0.5])) < 0.15


def test_reconstructed_cube_stays_in_world_frame(tmp_path: Path):
    cube = _cube_mesh(0.2, center=(0.1, 0.0, 0.6))
    target = np.array([0.1, 0.0, 0.6])
    K = np.array([[350.0, 0, 160], [0, 350.0, 120], [0, 0, 1]])
    w, h = 320, 240

    depth_maps, Ks, T_wcs = [], [], []
    for i, ang in enumerate([20, 70, 120]):
        T_wc = _camera_pose(0.9, ang, target)
        depth = render_mesh_depth(cube.copy().apply_transform(invert_T(T_wc)), K, w, h)
        dpath = tmp_path / f"d{i}.npy"
        np.save(dpath, depth.astype(np.float32))
        depth_maps.append(depth)
        Ks.append(K)
        T_wcs.append(T_wc)

    bounds = robust_expand_bounds(compute_bounds_from_depth_points(depth_maps, Ks, T_wcs), 0.05)
    tsdf = PlainTSDFReconstructor(PlainTSDFConfig(voxel_length_m=0.005, integrate_color=False), bounds)
    for i, ang in enumerate([20, 70, 120]):
        T_wc = _camera_pose(0.9, ang, target)
        tsdf.integrate_view(None, tmp_path / f"d{i}.npy", K, T_cam_world=invert_T(T_wc))

    mesh = o3d_mesh_to_trimesh(tsdf.extract_mesh())
    centroid = mesh.vertices.mean(axis=0)
    assert np.linalg.norm(centroid - target) < 0.08


def test_no_gt_depth_cheating(tmp_path: Path):
    """Inference must not require GT depth path."""
    K = np.eye(3)
    K[0, 0] = K[1, 1] = 500
    K[0, 2], K[1, 2] = 160, 120
    depth = np.ones((240, 320), dtype=np.float64) * 0.8
    perfect_path = tmp_path / "perfect.npy"
    np.save(perfect_path, depth.astype(np.float32))

    view = ViewRecord(
        dataset="bop_tless",
        scene_id="s1",
        view_id="v1",
        left_path="left.png",
        right_path="right.png",
        gt_depth_path="/invalid/gt_depth.png",
        K=K,
        stereo=StereoCalibration(has_true_stereo=True, baseline_m=0.06, rectified=True, synthetic=True),
        synthetic=True,
    )
    est = PerfectDepthEstimator({("s1", "v1"): perfect_path})
    pred = est.predict_view(view, tmp_path / "left.png", tmp_path / "right.png", tmp_path / "out")
    assert pred.depth_m.shape == depth.shape
    assert np.allclose(pred.depth_m, depth, atol=1e-5)


def test_no_stereo_raises_clear_error(tmp_path: Path):
    ckpt = tmp_path / "model.pth"
    ckpt.touch()
    (tmp_path / "cfg.yaml").write_text("vit_size: vitl\n", encoding="utf-8")
    view = ViewRecord(
        dataset="bop_tless",
        scene_id="000001",
        view_id="000000",
        rgb_path="rgb.png",
        gt_depth_path="gt_depth.png",
        K=np.eye(3),
        stereo=StereoCalibration(has_true_stereo=False, source="bop_standard"),
    )
    cfg = FoundationStereoConfig(
        foundationstereo_repo=tmp_path,
        ckpt=ckpt,
    )
    wrapper = FoundationStereoWrapper(cfg)
    with pytest.raises(ValueError, match="No true stereo pair"):
        wrapper.validate_view(view)


def test_bop_object_centric_pose_convention():
    T_model_cam = make_T(np.eye(3), np.array([0.0, 0.0, 0.5]))
    T_cam_model = invert_T(T_model_cam)
    view = ViewRecord(
        dataset="bop_tless",
        scene_id="s",
        view_id="v",
        left_path="l.png",
        right_path="r.png",
        K=np.eye(3),
        stereo=StereoCalibration(has_true_stereo=True, baseline_m=0.06, synthetic=True),
        object_poses=[
            ObjectPoseRecord(
                obj_id=1,
                instance_id=0,
                T_model_cam=T_model_cam,
                T_cam_model=T_cam_model,
                model_path="m.ply",
            )
        ],
    )
    T_wc, T_cw = world_cam_from_view(view, object_centric=True)
    assert np.allclose(T_wc, T_cam_model)
    assert np.allclose(T_cw, T_model_cam)
    assert np.allclose(extrinsic_for_open3d(T_wc, T_cw), T_model_cam)


def test_synthetic_stereo_pair_depth_differs():
    mesh = _cube_mesh()
    K = np.array([[400.0, 0, 160], [0, 400.0, 120], [0, 0, 1]])
    out = render_synthetic_stereo_pair([mesh], [np.eye(4)], K, 320, 240, 0.06)
    assert out["synthetic"] is True
    assert not np.allclose(out["left_depth_m"], out["right_depth_m"])

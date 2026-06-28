"""Tests for BOP dataset adapter."""

from pathlib import Path

import cv2
import json
import numpy as np
import trimesh

from volume_benchmark.datasets.bop_adapter import convert_bop_scene_sample, prepare_bop_scan


def _write_bop_camera(path: Path, K: np.ndarray, R: np.ndarray, t_mm: np.ndarray) -> None:
    payload = {
        "cam_K": K.reshape(-1).tolist(),
        "cam_R_w2c": R.reshape(-1).tolist(),
        "cam_t_w2c": t_mm.reshape(-1).tolist(),
    }
    path.write_text(json.dumps(payload))


def test_convert_bop_scene_sample(tmp_path: Path):
    K = np.array([[500, 0, 160], [0, 500, 120], [0, 0, 1]], dtype=float)
    R = np.eye(3)
    t_mm = np.array([0.0, 0.0, 500.0])
    cam_path = tmp_path / "scene_camera.json"
    _write_bop_camera(cam_path, K, R, t_mm)

    depth_path = tmp_path / "depth.png"
    mask_path = tmp_path / "mask.png"
    depth_mm = np.full((240, 320), 500, dtype=np.uint16)
    cv2.imwrite(str(depth_path), depth_mm)
    cv2.imwrite(str(mask_path), np.full((240, 320), 255, dtype=np.uint8))

    K_out, frame = convert_bop_scene_sample(depth_path, mask_path, cam_path)
    assert np.allclose(K_out, K)
    assert frame.depth_m.dtype == np.float32
    assert abs(frame.depth_m[120, 160] - 0.5) < 1e-5
    assert frame.mask[120, 160]
    assert frame.T_cam_to_object.shape == (4, 4)


def test_prepare_bop_scan(tmp_path: Path):
    mesh = trimesh.creation.icosphere(radius=0.05)
    mesh_path = tmp_path / "model.ply"
    mesh.export(mesh_path)

    K = np.array([[400, 0, 64], [0, 400, 64], [0, 0, 1]], dtype=float)
    R = np.eye(3)
    t_mm = np.array([0.0, 0.0, 400.0])

    frames_data = []
    for i in range(2):
        d = tmp_path / f"depth_{i}.png"
        m = tmp_path / f"mask_{i}.png"
        c = tmp_path / f"cam_{i}.json"
        cv2.imwrite(str(d), np.full((128, 128), 400, dtype=np.uint16))
        cv2.imwrite(str(m), np.full((128, 128), 255, dtype=np.uint8))
        _write_bop_camera(c, K, R, t_mm)
        frames_data.append((d, m, c))

    out = prepare_bop_scan(tmp_path / "scan", mesh_path, frames_data, mesh_units="m")
    assert (out / "K.npy").exists()
    assert (out / "gt_mesh.ply").exists()
    assert (out / "gt_volume.json").exists()
    assert (out / "frames" / "frame_000_depth.npy").exists()

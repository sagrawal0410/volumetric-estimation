"""Synthetic shape generation for smoke tests."""

from __future__ import annotations

import numpy as np
import trimesh


def make_cube(side_m: float = 1.0) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[side_m, side_m, side_m])


def make_box(dims: tuple[float, float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=list(dims))


def make_sphere(radius: float = 0.5, subdivisions: int = 3) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def make_cylinder(radius: float = 0.3, height: float = 1.0) -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=radius, height=height)


def make_u_shape() -> trimesh.Trimesh:
    """Concave U-shaped mesh built from boxes (no boolean backend required)."""
    left = trimesh.creation.box(extents=[0.25, 0.5, 0.5])
    left.apply_translation([-0.375, 0, 0])
    right = trimesh.creation.box(extents=[0.25, 0.5, 0.5])
    right.apply_translation([0.375, 0, 0])
    bottom = trimesh.creation.box(extents=[0.75, 0.5, 0.15])
    bottom.apply_translation([0, 0, -0.175])
    return trimesh.util.concatenate([left, right, bottom])


def make_partial_cube(side: float = 1.0, keep_fraction: float = 0.6) -> trimesh.Trimesh:
    mesh = make_cube(side)
    verts = mesh.vertices
    mask = verts[:, 0] > -side / 2 + side * (1 - keep_fraction)
    face_mask = mask[mesh.faces].all(axis=1)
    return mesh.submesh([face_mask], append=True)


def mesh_to_noisy_cloud(mesh: trimesh.Trimesh, n: int = 5000, noise_std: float = 0.002) -> np.ndarray:
    rng = np.random.default_rng(42)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    pts = pts + rng.normal(0, noise_std, pts.shape)
    return pts


def analytical_volume(name: str, **kwargs) -> float:
    if name == "cube":
        s = kwargs.get("side_m", 1.0)
        return s ** 3
    if name == "box":
        d = kwargs.get("dims", (1, 1, 1))
        return d[0] * d[1] * d[2]
    if name == "sphere":
        r = kwargs.get("radius", 0.5)
        return 4 / 3 * np.pi * r ** 3
    if name == "cylinder":
        r = kwargs.get("radius", 0.3)
        h = kwargs.get("height", 1.0)
        return np.pi * r ** 2 * h
    raise ValueError(f"Unknown shape: {name}")

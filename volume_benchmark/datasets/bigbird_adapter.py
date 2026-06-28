"""Adapter for Berkeley BigBIRD-style object scans."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import cv2
import numpy as np
import trimesh
import yaml

from volume_benchmark.common.geometry import invert_T, transform_points
from volume_benchmark.common.io import Frame, save_prepared_scan
from volume_benchmark.common.mesh_volume import (
    GtType,
    clean_mesh,
    compute_mesh_volume_m3,
    load_mesh_as_meters,
)
from volume_benchmark.common.view_selection import (
    CandidateFrame,
    select_diverse_views,
)

PoseFormat = Literal["cam_to_object", "object_to_cam"]
GtSource = Literal["mesh", "merged_pointcloud", "mesh_then_merged_pointcloud"]

MESH_PATTERNS = ("*mesh*.ply", "*reconstruct*.ply", "*poisson*.ply", "*.obj")
MERGED_CLOUD_PATTERNS = ("*merged*.pcd", "*merged*.ply", "*cloud*.pcd", "*cloud*.ply")
VIEW_ID_RE = re.compile(r"(\d+)")


@dataclass
class BigBirdConfig:
    """Optional overrides for heterogeneous BigBIRD export layouts."""

    mesh_path: Optional[str] = None
    merged_pointcloud_path: Optional[str] = None
    depth_glob: Optional[str] = None
    mask_glob: Optional[str] = None
    pose_glob: Optional[str] = None
    pose_file: Optional[str] = None
    calibration_file: Optional[str] = None
    depth_scale_to_meters: float = 0.001
    pose_format: PoseFormat = "cam_to_object"
    pointcloud_view_glob: Optional[str] = None
    mesh_units: str = "auto"
    object_root: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> BigBirdConfig:
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"BigBIRD config must be a mapping: {path}")
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    @classmethod
    def merge(cls, base: BigBirdConfig, overrides: BigBirdConfig | None) -> BigBirdConfig:
        if overrides is None:
            return base
        data = {f: getattr(base, f) for f in base.__dataclass_fields__}
        for f in base.__dataclass_fields__:
            val = getattr(overrides, f)
            if val is not None and val != "":
                data[f] = val
        return cls(**data)


@dataclass
class BigBirdViewCandidate:
    """One discoverable BigBIRD observation."""

    view_id: str
    depth_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    pose_path: Optional[Path] = None
    pointcloud_path: Optional[Path] = None
    T_cam_to_object: Optional[np.ndarray] = None
    valid_pixels: int = 0
    source_info: dict = field(default_factory=dict)


@dataclass
class GroundTruthResult:
    gt_mesh_path: Path
    volume_m3: float
    gt_type: GtType
    watertight: bool
    source_path: Path
    pseudo_gt_method: Optional[str] = None
    exact_gt: bool = True


def _object_root_path(object_root: str | Path) -> Path:
    path = Path(object_root).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"BigBIRD object_root does not exist: {path}")
    return path


def _glob_recursive(root: Path, pattern: str) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and fnmatch.fnmatch(path.name.lower(), pattern.lower()):
            matches.append(path)
    return sorted(matches)


def _rank_mesh_candidate(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if "poisson" in name:
        return (0, name)
    if "reconstruct" in name:
        return (1, name)
    if "mesh" in name:
        return (2, name)
    return (3, name)


def discover_mesh_path(object_root: Path, config: BigBirdConfig) -> Optional[Path]:
    if config.mesh_path:
        path = Path(config.mesh_path)
        if not path.is_absolute():
            path = object_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Configured mesh_path not found: {path}")
        return path.resolve()

    candidates: list[Path] = []
    for pattern in MESH_PATTERNS:
        candidates.extend(_glob_recursive(object_root, pattern))
    candidates = [p for p in candidates if "merged" not in p.name.lower()]
    if not candidates:
        return None
    candidates.sort(key=_rank_mesh_candidate)
    return candidates[0].resolve()


def discover_merged_pointcloud_path(object_root: Path, config: BigBirdConfig) -> Optional[Path]:
    if config.merged_pointcloud_path:
        path = Path(config.merged_pointcloud_path)
        if not path.is_absolute():
            path = object_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Configured merged_pointcloud_path not found: {path}")
        return path.resolve()

    candidates: list[Path] = []
    for pattern in MERGED_CLOUD_PATTERNS:
        candidates.extend(_glob_recursive(object_root, pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name.lower())
    return candidates[0].resolve()


def load_bigbird_intrinsics(intrinsics_path: Path) -> np.ndarray:
    """Load a 3x3 camera matrix from .npy or whitespace-delimited text."""
    if intrinsics_path.suffix == ".npy":
        K = np.load(intrinsics_path)
        if K.shape != (3, 3):
            raise ValueError(f"Expected (3,3) K in {intrinsics_path}")
        return K.astype(np.float64)

    values: list[float] = []
    with intrinsics_path.open("r", encoding="utf-8") as f:
        for line in f:
            for token in line.split():
                if token.startswith("#"):
                    break
                values.append(float(token))
    arr = np.array(values, dtype=np.float64)
    if arr.size == 9:
        return arr.reshape(3, 3)
    if arr.size == 4:
        fx, fy, cx, cy = arr
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"Unsupported intrinsics format in {intrinsics_path}")


def discover_calibration_path(object_root: Path, config: BigBirdConfig) -> Optional[Path]:
    if config.calibration_file:
        path = Path(config.calibration_file)
        if not path.is_absolute():
            path = object_root / path
        if not path.is_file():
            raise FileNotFoundError(f"Configured calibration_file not found: {path}")
        return path.resolve()

    for pattern in ("K.npy", "*calib*.npy", "*intrinsic*.npy", "*calib*.txt", "*intrinsic*.txt"):
        hits = _glob_recursive(object_root, pattern)
        if hits:
            return hits[0].resolve()
    return None


def load_bigbird_pose(pose_path: Path) -> np.ndarray:
    """Load a 4x4 pose matrix from .npy or text."""
    if pose_path.suffix == ".npy":
        T = np.load(pose_path)
    else:
        rows = []
        with pose_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    rows.append([float(x) for x in line.split()])
        T = np.array(rows, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Pose must be 4x4 in {pose_path}")
    return T


def _normalize_pose(T: np.ndarray, pose_format: PoseFormat) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if pose_format == "object_to_cam":
        return invert_T(T)
    return T


def _view_id_from_path(path: Path) -> str:
    match = VIEW_ID_RE.search(path.stem)
    return match.group(1) if match else path.stem


def _match_by_view_id(paths: list[Path]) -> dict[str, Path]:
    return {_view_id_from_path(p): p for p in paths}


def count_valid_depth_pixels(depth_m: np.ndarray, mask: np.ndarray) -> int:
    valid = mask & np.isfinite(depth_m) & (depth_m > 0.01)
    return int(valid.sum())


def load_depth_meters(depth_path: Path, depth_scale: float) -> np.ndarray:
    if depth_path.suffix == ".npy":
        depth = np.load(depth_path)
        depth_m = depth.astype(np.float32)
        if depth_scale != 1.0:
            depth_m = depth_m * np.float32(depth_scale)
        return depth_m

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    return depth_raw.astype(np.float32) * np.float32(depth_scale)


def load_pointcloud_meters(path: Path, source_units: str = "auto") -> np.ndarray:
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix == ".pcd":
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points, dtype=np.float64)
    else:
        loaded = trimesh.load(str(path), process=False)
        if isinstance(loaded, trimesh.PointCloud):
            points = np.asarray(loaded.vertices, dtype=np.float64)
        elif isinstance(loaded, trimesh.Trimesh):
            points = np.asarray(loaded.vertices, dtype=np.float64)
        else:
            raise TypeError(f"Unsupported point cloud type in {path}: {type(loaded)}")

    if points.size == 0:
        raise ValueError(f"Point cloud is empty: {path}")

    if source_units == "auto":
        extent = float(np.max(points.max(axis=0) - points.min(axis=0)))
        source_units = "mm" if extent > 5.0 else "m"
    if source_units == "mm":
        points = points * 0.001
    elif source_units != "m":
        raise ValueError(f"Unsupported point cloud units: {source_units}")
    return points


def rasterize_points_to_depth(
    points_object: np.ndarray,
    K: np.ndarray,
    T_cam_to_object: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer rasterize object-frame points into depth (m) and bool mask."""
    height, width = image_shape
    T_object_to_cam = invert_T(T_cam_to_object)
    points_cam = transform_points(points_object, T_object_to_cam)
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    in_front = z > 1e-4
    if not np.any(in_front):
        depth = np.zeros((height, width), dtype=np.float32)
        mask = np.zeros((height, width), dtype=bool)
        return depth, mask

    x, y, z = x[in_front], y[in_front], z[in_front]
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    in_image = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui, vi, z = ui[in_image], vi[in_image], z[in_image]

    depth = np.full((height, width), np.inf, dtype=np.float32)
    for uu, vv, zz in zip(ui, vi, z, strict=False):
        if zz < depth[vv, uu]:
            depth[vv, uu] = zz
    mask = np.isfinite(depth) & (depth < np.inf)
    depth[~mask] = 0.0
    return depth, mask


def _open3d_available() -> bool:
    if os.environ.get("VOLUME_BENCHMARK_SKIP_OPEN3D", "0") == "1":
        return False
    try:
        import open3d as o3d  # noqa: F401
        return True
    except Exception:
        return False


def _estimate_alpha_shape_mesh(points: np.ndarray, alpha: float = 0.02) -> Optional[trimesh.Trimesh]:
    if points.shape[0] < 100 or not _open3d_available():
        return None
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    except Exception:
        return None
    if len(mesh_o3d.triangles) == 0:
        return None
    verts = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if len(mesh.faces) == 0:
        return None
    return clean_mesh(mesh)


def _estimate_poisson_mesh(points: np.ndarray, depth: int = 8) -> Optional[trimesh.Trimesh]:
    if points.shape[0] < 200 or not _open3d_available():
        return None
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )
    except Exception:
        return None
    if len(mesh_o3d.triangles) == 0:
        return None
    densities = np.asarray(densities)
    if densities.size:
        keep = densities > np.quantile(densities, 0.02)
        mesh_o3d = mesh_o3d.select_by_index(np.where(keep)[0])
    verts = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    if len(faces) == 0:
        return None
    return clean_mesh(trimesh.Trimesh(vertices=verts, faces=faces, process=True))


def estimate_pseudo_gt_from_pointcloud(
    points_m: np.ndarray,
    voxel_size: float,
    out_mesh_path: Path,
) -> tuple[float, trimesh.Trimesh, str]:
    """
    Estimate pseudo-GT volume from a dense reconstruction point cloud.

    Returns (volume_m3, mesh, method_name).
    """
    if points_m.shape[0] < 50:
        raise ValueError("Point cloud too sparse for pseudo-GT volume estimation")

    mesh = _estimate_alpha_shape_mesh(points_m)
    method = "alpha_shape"
    if mesh is not None and mesh.is_watertight and mesh.volume > 0:
        volume = abs(float(mesh.volume))
        mesh.export(out_mesh_path)
        return volume, mesh, method

    mesh = _estimate_poisson_mesh(points_m)
    method = "poisson"
    if mesh is not None and mesh.is_watertight and mesh.volume > 0:
        volume = abs(float(mesh.volume))
        mesh.export(out_mesh_path)
        return volume, mesh, method

    method = "voxel_occupancy"
    voxel_volume = _voxel_occupancy_volume(points_m, voxel_size)
    if voxel_volume <= 0:
        raise ValueError("Voxel pseudo-GT volume is non-positive")

    hull = _points_to_proxy_mesh(points_m)
    hull.export(out_mesh_path)
    return float(voxel_volume), hull, method


def _points_to_proxy_mesh(points_m: np.ndarray) -> trimesh.Trimesh:
    try:
        hull = trimesh.Trimesh(vertices=points_m, faces=[]).convex_hull
        if len(hull.faces) > 0:
            return hull
    except Exception:
        pass
    lo = points_m.min(axis=0)
    hi = points_m.max(axis=0)
    extents = np.maximum(hi - lo, 1e-4)
    box = trimesh.creation.box(extents=extents)
    box.apply_translation((lo + hi) / 2.0)
    return box


def _voxel_occupancy_volume(points_m: np.ndarray, voxel_size: float) -> float:
    """Count occupied voxels for a point cloud without requiring Open3D."""
    if points_m.shape[0] == 0:
        return 0.0
    mins = points_m.min(axis=0)
    coords = np.floor((points_m - mins) / voxel_size).astype(np.int64)
    unique = np.unique(coords, axis=0)
    return float(unique.shape[0] * (voxel_size ** 3))


def resolve_ground_truth(
    object_root: Path,
    config: BigBirdConfig,
    out_dir: Path,
    gt_source: GtSource,
    gt_voxel_size: float,
    repair_mesh: bool = True,
) -> GroundTruthResult:
    """Resolve GT or pseudo-GT mesh and volume for a BigBIRD object folder."""
    mesh_path = discover_mesh_path(object_root, config)
    cloud_path = discover_merged_pointcloud_path(object_root, config)
    gt_mesh_out = out_dir / "gt_mesh.ply"

    def _from_mesh(path: Path) -> Optional[GroundTruthResult]:
        mesh = load_mesh_as_meters(path, source_units=config.mesh_units)
        try:
            volume, watertight, gt_type = compute_mesh_volume_m3(mesh, repair=False)
            exact = gt_type == "mesh_watertight"
            shutil.copy2(path, gt_mesh_out)
            return GroundTruthResult(
                gt_mesh_path=gt_mesh_out,
                volume_m3=volume,
                gt_type=gt_type,
                watertight=watertight,
                source_path=path,
                exact_gt=exact,
            )
        except ValueError:
            if not repair_mesh:
                return None
        try:
            volume, watertight, gt_type = compute_mesh_volume_m3(mesh, repair=True)
            cleaned = clean_mesh(mesh)
            cleaned.export(gt_mesh_out)
            return GroundTruthResult(
                gt_mesh_path=gt_mesh_out,
                volume_m3=volume,
                gt_type=gt_type,
                watertight=watertight,
                source_path=path,
                exact_gt=False,
            )
        except ValueError:
            if gt_source == "mesh":
                raise ValueError(
                    f"Mesh {path} is not watertight and could not be repaired; "
                    "refusing to treat as exact ground truth."
                )
            return None

    if gt_source in ("mesh", "mesh_then_merged_pointcloud") and mesh_path is not None:
        result = _from_mesh(mesh_path)
        if result is not None:
            return result

    if gt_source in ("merged_pointcloud", "mesh_then_merged_pointcloud"):
        if cloud_path is None:
            raise FileNotFoundError(
                f"No merged point cloud found under {object_root}. "
                "Expected patterns like *merged*.pcd or configure merged_pointcloud_path."
            )
        points = load_pointcloud_meters(cloud_path, source_units=config.mesh_units)
        volume, _, method = estimate_pseudo_gt_from_pointcloud(
            points, gt_voxel_size, gt_mesh_out
        )
        return GroundTruthResult(
            gt_mesh_path=gt_mesh_out,
            volume_m3=volume,
            gt_type="full_reconstruction_pseudo_gt",
            watertight=False,
            source_path=cloud_path,
            pseudo_gt_method=method,
            exact_gt=False,
        )

    raise FileNotFoundError(
        f"Could not resolve ground truth under {object_root} with gt_source={gt_source!r}"
    )


def discover_view_candidates(object_root: Path, config: BigBirdConfig) -> list[BigBirdViewCandidate]:
    """Discover per-view depth/mask/pose or point-cloud observations."""
    depth_glob = config.depth_glob or "*depth*.png"
    mask_glob = config.mask_glob or "*mask*.png"
    pose_glob = config.pose_glob or "*pose*.npy"
    pc_glob = config.pointcloud_view_glob or "*view*.pcd"

    depth_paths = _glob_recursive(object_root, depth_glob)
    if not depth_paths:
        depth_paths = _glob_recursive(object_root, "*depth*.npy")
    mask_paths = _glob_recursive(object_root, mask_glob)
    pose_paths = _glob_recursive(object_root, pose_glob)
    if not pose_paths:
        pose_paths = _glob_recursive(object_root, "*T_cam*.npy")
    pc_paths = _glob_recursive(object_root, pc_glob)

    depth_by_id = _match_by_view_id(depth_paths)
    mask_by_id = _match_by_view_id(mask_paths)
    pose_by_id = _match_by_view_id(pose_paths)
    pc_by_id = _match_by_view_id(pc_paths)

    all_ids = sorted(
        set(depth_by_id) | set(mask_by_id) | set(pose_by_id) | set(pc_by_id),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    candidates: list[BigBirdViewCandidate] = []
    for view_id in all_ids:
        candidates.append(
            BigBirdViewCandidate(
                view_id=view_id,
                depth_path=depth_by_id.get(view_id),
                mask_path=mask_by_id.get(view_id),
                pose_path=pose_by_id.get(view_id),
                pointcloud_path=pc_by_id.get(view_id),
            )
        )
    return candidates


def _load_pose_for_candidate(
    candidate: BigBirdViewCandidate,
    config: BigBirdConfig,
    object_root: Path,
) -> np.ndarray:
    if candidate.pose_path is not None:
        T = load_bigbird_pose(candidate.pose_path)
        return _normalize_pose(T, config.pose_format)
    if config.pose_file:
        pose_file = Path(config.pose_file)
        if not pose_file.is_absolute():
            pose_file = object_root / pose_file
        all_poses = np.load(pose_file)
        idx = int(candidate.view_id) if candidate.view_id.isdigit() else 0
        if all_poses.ndim == 3:
            T = all_poses[idx]
        else:
            T = all_poses
        return _normalize_pose(T, config.pose_format)
    raise FileNotFoundError(
        f"No pose available for view {candidate.view_id}. "
        "Provide pose_glob or pose_file in bigbird_config.yaml."
    )


def _score_view_candidate(
    candidate: BigBirdViewCandidate,
    config: BigBirdConfig,
    K: np.ndarray,
    image_shape: tuple[int, int],
    object_root: Path,
) -> BigBirdViewCandidate:
    depth_m: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None

    if candidate.depth_path is not None and candidate.mask_path is not None:
        depth_m = load_depth_meters(candidate.depth_path, config.depth_scale_to_meters)
        mask_raw = cv2.imread(str(candidate.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise FileNotFoundError(f"Could not read mask: {candidate.mask_path}")
        mask = mask_raw > 0
        if depth_m.shape != mask.shape:
            raise ValueError(
                f"Depth/mask shape mismatch for view {candidate.view_id}: "
                f"{depth_m.shape} vs {mask.shape}"
            )
    elif candidate.pointcloud_path is not None:
        T = _load_pose_for_candidate(candidate, config, object_root)
        candidate.T_cam_to_object = T
        points = load_pointcloud_meters(candidate.pointcloud_path, config.mesh_units)
        depth_m, mask = rasterize_points_to_depth(points, K, T, image_shape)
    else:
        candidate.valid_pixels = 0
        return candidate

    candidate.valid_pixels = count_valid_depth_pixels(depth_m, mask)
    if candidate.T_cam_to_object is None:
        candidate.T_cam_to_object = _load_pose_for_candidate(candidate, config, object_root)
    return candidate


def build_frame_from_candidate(
    candidate: BigBirdViewCandidate,
    config: BigBirdConfig,
    K: np.ndarray,
    image_shape: tuple[int, int],
) -> Frame:
    """Build a normalized Frame from a scored BigBIRD view candidate."""
    if candidate.T_cam_to_object is None:
        raise ValueError(f"View {candidate.view_id} is missing T_cam_to_object")

    depth_m: np.ndarray
    mask: np.ndarray
    supports_voxel_carving = True
    points_object_path: Optional[str] = None

    if candidate.depth_path is not None and candidate.mask_path is not None:
        depth_m = load_depth_meters(candidate.depth_path, config.depth_scale_to_meters)
        mask_raw = cv2.imread(str(candidate.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise FileNotFoundError(f"Could not read mask: {candidate.mask_path}")
        mask = mask_raw > 0
    elif candidate.pointcloud_path is not None:
        points = load_pointcloud_meters(candidate.pointcloud_path, config.mesh_units)
        depth_m, mask = rasterize_points_to_depth(
            points, K, candidate.T_cam_to_object, image_shape
        )
        supports_voxel_carving = False
        points_object_path = str(candidate.pointcloud_path)
    else:
        raise ValueError(f"View {candidate.view_id} has neither depth/mask nor point cloud")

    source_info = {
        "dataset": "bigbird",
        "view_id": candidate.view_id,
        "valid_pixels": candidate.valid_pixels,
        "supports_voxel_carving": supports_voxel_carving,
        "frame_backing": "depth" if candidate.depth_path is not None else "pointcloud",
    }
    if candidate.depth_path is not None:
        source_info["depth_path"] = str(candidate.depth_path)
    if candidate.mask_path is not None:
        source_info["mask_path"] = str(candidate.mask_path)
    if candidate.pose_path is not None:
        source_info["pose_path"] = str(candidate.pose_path)
    if points_object_path is not None:
        source_info["pointcloud_path"] = points_object_path

    return Frame(
        depth_m=depth_m,
        mask=mask,
        T_cam_to_object=candidate.T_cam_to_object,
        source_info=source_info,
    )


def prepare_bigbird_scan(
    object_root: str | Path,
    out_dir: str | Path,
    num_views: int = 5,
    config_path: Optional[str] = None,
    min_valid_depth_pixels: int = 1000,
    gt_source: GtSource = "mesh_then_merged_pointcloud",
    gt_voxel_size: float = 0.0015,
    repair_mesh: bool = True,
) -> Path:
    """
    Prepare a normalized scan from one BigBIRD object folder.

    Auto-discovers reconstructed mesh or merged point cloud for GT/pseudo-GT,
    selects angularly diverse RGB-D views, and writes the prepared scan format.
    """
    root = _object_root_path(object_root)
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    config = BigBirdConfig(object_root=str(root))
    if config_path:
        config = BigBirdConfig.merge(config, BigBirdConfig.from_yaml(config_path))

    calib_path = discover_calibration_path(root, config)
    if calib_path is None:
        raise FileNotFoundError(
            f"No calibration file found under {root}. "
            "Add calibration_file to bigbird_config.yaml."
        )
    K = load_bigbird_intrinsics(calib_path)

    gt = resolve_ground_truth(
        root, config, out, gt_source=gt_source, gt_voxel_size=gt_voxel_size, repair_mesh=repair_mesh
    )

    candidates = discover_view_candidates(root, config)
    if not candidates:
        raise FileNotFoundError(
            f"No per-view depth/mask/pose data discovered under {root}. "
            "Configure depth_glob/mask_glob/pose_glob in bigbird_config.yaml."
        )

    sample = next(
        (c for c in candidates if c.depth_path is not None or c.pointcloud_path is not None),
        None,
    )
    if sample is None:
        raise FileNotFoundError("No depth or point-cloud views found")

    if sample.depth_path is not None:
        sample_depth = load_depth_meters(sample.depth_path, config.depth_scale_to_meters)
        image_shape = sample_depth.shape
    else:
        image_shape = (480, 640)

    scored: list[BigBirdViewCandidate] = []
    for cand in candidates:
        try:
            if cand.depth_path is None and cand.pointcloud_path is None:
                continue
            if cand.depth_path is not None and cand.mask_path is None:
                continue
            cand = _score_view_candidate(cand, config, K, image_shape, root)
            if cand.T_cam_to_object is None:
                cand.T_cam_to_object = _load_pose_for_candidate(cand, config, root)
            scored.append(cand)
        except (FileNotFoundError, ValueError):
            continue

    if not scored:
        raise ValueError("No valid BigBIRD views could be scored")

    candidate_frames: list[CandidateFrame] = []
    for cand in scored:
        frame = build_frame_from_candidate(cand, config, K, image_shape)
        candidate_frames.append(
            CandidateFrame(
                depth_m=frame.depth_m,
                mask=frame.mask,
                K=K,
                T_cam_to_object=frame.T_cam_to_object,
                metadata={
                    "valid_object_depth_pixels": cand.valid_pixels,
                    "frame_id": cand.view_id,
                    "scene_id": str(root.name),
                    **frame.source_info,
                },
            )
        )

    selected_frames = select_diverse_views(
        candidate_frames,
        num_views=num_views,
        min_valid_depth_pixels=min_valid_depth_pixels,
        min_angle_deg=25.0,
        prefer_high_elevation=True,
        selected_views_path=out / "selected_views.json",
    )
    selected_ids = [c.frame_id for c in selected_frames]
    scored_by_id = {c.view_id: c for c in scored}
    selected = [scored_by_id[view_id] for view_id in selected_ids]
    frames = [build_frame_from_candidate(c, config, K, image_shape) for c in selected]

    metadata: dict[str, Any] = {
        "dataset": "bigbird",
        "object_root": str(root),
        "num_source_views": len(scored),
        "num_selected_views": len(frames),
        "selected_view_ids": [c.view_id for c in selected],
        "calibration_file": str(calib_path),
        "gt_source": gt_source,
        "exact_gt": gt.exact_gt,
        "supports_voxel_carving": all(
            f.source_info.get("supports_voxel_carving", True) for f in frames
        ),
    }
    if gt.pseudo_gt_method:
        metadata["pseudo_gt_method"] = gt.pseudo_gt_method

    save_prepared_scan(out, K, frames, gt.gt_mesh_path, metadata=metadata)

    gt_payload: dict[str, Any] = {
        "volume_m3": gt.volume_m3,
        "volume_cm3": gt.volume_m3 * 1e6,
        "gt_type": gt.gt_type,
        "watertight": gt.watertight,
        "source_mesh": str(gt.source_path),
        "exact_gt": gt.exact_gt,
    }
    if gt.pseudo_gt_method:
        gt_payload["pseudo_gt_method"] = gt.pseudo_gt_method
    with (out / "gt_volume.json").open("w", encoding="utf-8") as f:
        json.dump(gt_payload, f, indent=2)

    return out


def load_bigbird_config(config_path: str | Path) -> BigBirdConfig:
    """Load a BigBIRD YAML config file."""
    return BigBirdConfig.from_yaml(config_path)

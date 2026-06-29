"""Configuration dataclasses and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DenoiseConfig:
    statistical_nb_neighbors: int = 20
    statistical_std_ratio: float = 2.0
    radius_nb_points: int = 16
    radius_search_m: float = 0.05
    voxel_downsample_m: float | None = None
    enable_statistical: bool = True
    enable_radius: bool = True


@dataclass
class PlaneRemovalConfig:
    distance_threshold_m: float = 0.01
    ransac_n: int = 3
    num_iterations: int = 1000
    min_plane_inlier_ratio: float = 0.05


@dataclass
class ClusteringConfig:
    eps_m: float = 0.05
    min_points: int = 50
    choose_cluster: str = "largest"  # largest, nearest_to_roi_center, all


@dataclass
class SegmentationConfig:
    mode: str = "none"
    height_above_plane_threshold_m: float = 0.02
    cluster_id: int | None = None


@dataclass
class PoissonConfig:
    depth: int = 9
    width: int = 0
    scale: float = 1.1
    linear_fit: bool = False
    density_quantile: float = 0.01


@dataclass
class BallPivotingConfig:
    radius_multipliers: list[float] = field(default_factory=lambda: [1.5, 2.0, 3.0, 4.0])


@dataclass
class AlphaShapeConfig:
    alpha_values: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.1, 0.2])


@dataclass
class VoxelConfig:
    voxel_sizes_m: list[float] = field(default_factory=lambda: [0.002, 0.003, 0.005, 0.01])


@dataclass
class HeightfieldConfig:
    grid_resolution_m: float = 0.01
    height_stat: str = "p95"  # max, p95, p90, mean_top_k
    min_points_per_cell: int = 1
    hole_fill_method: str = "nearest"  # nearest, interpolate, none
    base_height_m: float = 0.0


@dataclass
class MeshCleaningConfig:
    merge_vertices_tolerance_m: float = 1e-6
    min_component_faces: int = 10
    keep_largest_component: bool = True
    fill_small_holes: bool = True


@dataclass
class MeshRepairConfig:
    use_pymeshlab: bool = True
    fix_normals: bool = True
    fill_holes: bool = True
    fix_inversion: bool = True
    fix_winding: bool = True


@dataclass
class ConsensusConfig:
    pile_mode: bool = False
    estimator_priority: list[str] = field(
        default_factory=lambda: [
            "direct_mesh_volume",
            "repaired_mesh_volume",
            "voxel_occupancy_volume",
            "alpha_shape_volume",
            "convex_hull_volume",
        ]
    )
    pile_warning: str = ""


@dataclass
class OutputConfig:
    save_intermediates: bool = True
    generate_html_report: bool = True
    generate_screenshots: bool = True
    seed: int = 42


@dataclass
class PipelineConfig:
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    plane_removal: PlaneRemovalConfig = field(default_factory=PlaneRemovalConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    poisson: PoissonConfig = field(default_factory=PoissonConfig)
    ball_pivoting: BallPivotingConfig = field(default_factory=BallPivotingConfig)
    alpha_shape: AlphaShapeConfig = field(default_factory=AlphaShapeConfig)
    voxel: VoxelConfig = field(default_factory=VoxelConfig)
    heightfield: HeightfieldConfig = field(default_factory=HeightfieldConfig)
    mesh_cleaning: MeshCleaningConfig = field(default_factory=MeshCleaningConfig)
    mesh_repair: MeshRepairConfig = field(default_factory=MeshRepairConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mode: str = "object_or_pile"
    reconstruct_mesh_from_cloud: bool = True
    run_poisson: bool = True
    run_bpa: bool = True
    run_alpha_shape: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        def sub(key: str, cls_type: type) -> Any:
            if key in data and isinstance(data[key], dict):
                return cls_type(**{k: v for k, v in data[key].items() if k in cls_type.__dataclass_fields__})
            return cls_type()

        cfg = cls(
            denoise=sub("denoise", DenoiseConfig),
            plane_removal=sub("plane_removal", PlaneRemovalConfig),
            clustering=sub("clustering", ClusteringConfig),
            segmentation=sub("segmentation", SegmentationConfig),
            poisson=sub("poisson", PoissonConfig),
            ball_pivoting=sub("ball_pivoting", BallPivotingConfig),
            alpha_shape=sub("alpha_shape", AlphaShapeConfig),
            voxel=sub("voxel", VoxelConfig),
            heightfield=sub("heightfield", HeightfieldConfig),
            mesh_cleaning=sub("mesh_cleaning", MeshCleaningConfig),
            mesh_repair=sub("mesh_repair", MeshRepairConfig),
            consensus=sub("consensus", ConsensusConfig),
            output=sub("output", OutputConfig),
        )
        for k in ("mode", "reconstruct_mesh_from_cloud", "run_poisson", "run_bpa", "run_alpha_shape"):
            if k in data:
                setattr(cfg, k, data[k])
        return cfg

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return PipelineConfig.from_dict(data)


def save_config(config: PipelineConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

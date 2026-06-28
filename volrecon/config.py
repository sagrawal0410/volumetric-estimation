"""Global configuration defaults for volrecon."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DEFAULT_MANIFEST_DIR = DEFAULT_PROCESSED_ROOT / "manifests"

# BOP datasets typically use millimeters for models and translations.
BOP_DEFAULT_UNITS = "mm"
INTERNAL_UNITS = "m"

# Synthetic stereo defaults (sanity-check mode only).
DEFAULT_SYNTHETIC_BASELINE_M = 0.06
DEFAULT_VOXEL_SIZE_M = 0.005

# Modality names used across manifests.
MODALITY_RGB = "rgb"
MODALITY_LEFT = "left"
MODALITY_RIGHT = "right"
MODALITY_MONO = "mono"
MODALITY_MONO_STEREO = "mono_stereo"
MODALITY_GT_DEPTH = "gt_depth"
MODALITY_MASK = "mask"
MODALITY_ESTIMATED_DEPTH = "estimated_depth"

INFERENCE_MODALITIES = frozenset(
    {MODALITY_RGB, MODALITY_LEFT, MODALITY_RIGHT, MODALITY_MONO, MODALITY_MONO_STEREO}
)
EVAL_ONLY_MODALITIES = frozenset({MODALITY_GT_DEPTH, MODALITY_MASK})


@dataclass
class PreprocessConfig:
    """Runtime options shared by dataset extractors."""

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    processed_root: Path = field(default_factory=lambda: DEFAULT_PROCESSED_ROOT)
    symlink: bool = True
    overwrite: bool = False
    synthetic_baseline_m: float = DEFAULT_SYNTHETIC_BASELINE_M
    voxel_size_m: float = DEFAULT_VOXEL_SIZE_M

    def resolve_path(self, path: Path | str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

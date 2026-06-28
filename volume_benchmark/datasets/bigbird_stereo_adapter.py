"""BigBIRD stereo adapter — rendered mode only (BigBIRD adapter removed from main pipeline)."""

from __future__ import annotations

from pathlib import Path


def prepare_bigbird_stereo_rendered(
    object_root: str | Path,
    out_dir: str | Path,
    baseline_m: float = 0.12,
    num_views: int = 5,
) -> Path:
    root = Path(object_root)
    mesh_candidates = list(root.rglob("*mesh*.ply")) + list(root.rglob("*poisson*.ply"))
    if not mesh_candidates:
        raise FileNotFoundError(
            f"No mesh found under {object_root}. BigBIRD stereo requires a reconstructed mesh. "
            "Real calibrated stereo pairs are not auto-discovered; add bigbird_config.yaml "
            "with left/right paths for source_mode=real_dataset_stereo."
        )
    raise NotImplementedError(
        "BigBIRD stereo rendered preparation is not fully implemented. "
        "Use T-LESS or WildRGB-D for FoundationStereo benchmarking, or provide a "
        "prepared_stereo_scan manually."
    )

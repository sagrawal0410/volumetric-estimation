"""T-LESS BOP stereo rendered preparation."""

from __future__ import annotations

from pathlib import Path

from volume_benchmark.datasets.bop_stereo_adapter import prepare_bop_stereo_rendered


def prepare_tless_stereo_rendered(
    dataset_root: str | Path,
    split: str,
    object_id: int,
    out_dir: str | Path,
    baseline_m: float = 0.12,
    num_views: int = 5,
    min_visib_fract: float = 0.5,
) -> Path:
    return prepare_bop_stereo_rendered(
        dataset_root=dataset_root,
        split=split,
        object_id=object_id,
        out_dir=out_dir,
        baseline_m=baseline_m,
        num_views=num_views,
        min_visib_fract=min_visib_fract,
        dataset_name="tless",
    )
